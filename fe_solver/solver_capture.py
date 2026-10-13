"""Temporary drop-in solver capturing the first two t_n=0 KKT systems.

Replace solver.py with this file only for a diagnostic run. Captures go to
<output.directory>/kkt_capture/kkt_debug_t0_iter{0,1}.h5. The first attempted
increment is captured; cutback retries do not replace its systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import h5py
import numpy as np
from scipy import sparse

from .assembly import (
    AssemblyResult,
    FEModel,
    assemble_external,
    assemble_internal,
    commit_trial_states,
    observe_committed,
)
from .config import Deck, load_deck
from .constraints import ConstraintSystem
from .execution import ElementExecutor
from .io import HDF5ResultWriter, load_restart, write_restart, write_run_log
from .linear_solver import (
    LinearSolverConfig,
    close_factor,
    factor_kkt as _factor_kkt,
    solve_factor,
)
from .preprocess import PreparedAnalysis, prepare_analysis
from .reporting import (
    build_analysis_summary,
    print_linear_system_summary,
    print_startup_summary,
    record_sparse_system,
)
from .types import ModelError, RecoverableError

from contextlib import contextmanager
from collections.abc import MutableMapping
from typing import Generator


@contextmanager
def timer(registry: MutableMapping[str, float], key: str) -> Generator:
    start = perf_counter_ns()
    try:
        yield
    finally:
        registry[key] = (perf_counter_ns() - start) * 1e-9


@dataclass
class IncrementRecord:
    t_n: float
    t_np1: float
    attempts: int
    newton_iterations: int
    cutbacks: int


@dataclass
class AnalysisResult:
    model: FEModel
    constraints: ConstraintSystem
    t: float
    u: np.ndarray
    lambdas: np.ndarray
    increments: list[IncrementRecord] = field(default_factory=list)
    newton_history: list[dict] = field(default_factory=list)
    verification: dict[str, float] = field(default_factory=dict)
    analysis: dict[str, object] = field(default_factory=dict)
    execution: dict[str, object] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class _NewtonResult:
    u: np.ndarray
    lambdas: np.ndarray
    assembly: AssemblyResult
    iterations: int


def _inf_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector, ord=np.inf)) if vector.size else 0.0


def _scaled_residual_merit(
    r_u: np.ndarray,
    r_c: np.ndarray,
    force_tolerance: float,
    constraint_tolerance: float,
) -> float:
    """Measure both KKT residual blocks in their convergence-test units."""
    def ratio(residual: np.ndarray, tolerance: float) -> float:
        value = _inf_norm(residual)
        if tolerance > 0.0:
            return value / tolerance
        return 0.0 if value == 0.0 else np.inf

    return max(
        ratio(r_u, force_tolerance),
        ratio(r_c, constraint_tolerance),
    )


def _capture_kkt_system(
    model: FEModel,
    K: sparse.csr_matrix,
    C: sparse.csr_matrix,
    r_u: np.ndarray,
    r_c: np.ndarray,
    t_n: float,
    t_np1: float,
    iteration: int,
) -> Path:
    """Write the same sparse-system layout consumed by both native benchmarks."""
    output_dir = model.deck.resolve(str(model.deck.data["output"]["directory"]))
    path = output_dir / "kkt_capture" / f"kkt_debug_t0_iter{iteration}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(path, "x") as archive:
            group = archive.create_group(f"t_n={t_n}/iteration={iteration}")
            group.attrs["t_np1"] = t_np1
            for name, matrix in (("K", K), ("C", C)):
                matrix = matrix.tocsr()
                item = group.create_group(name)
                item.attrs["shape"] = matrix.shape
                for key in ("data", "indices", "indptr"):
                    item.create_dataset(key, data=getattr(matrix, key))
            group.create_dataset("r_u", data=np.asarray(r_u))
            group.create_dataset("r_c", data=np.asarray(r_c))
    except FileExistsError as exc:
        raise ModelError(
            f"capture already exists: {path}; use a fresh output directory"
        ) from exc
    print(f"captured KKT: {path}, group=t_n={t_n}/iteration={iteration}", flush=True)
    return path


def _newton_attempt(
    model: FEModel,
    constraints: ConstraintSystem,
    u_n: np.ndarray,
    lambda_n: np.ndarray,
    t_n: float,
    t_np1: float,
    history: list[dict],
    iteration_observer: Callable[[dict, AssemblyResult], None] | None = None,
    element_executor: ElementExecutor | None = None,
    debug_timing: bool = False,
    linear_system_observer: Callable[[sparse.csr_matrix], None] | None = None,
    linear_solver: LinearSolverConfig | None = None,
    capture_attempt: bool = False,
) -> _NewtonResult:
    linear_solver = linear_solver or LinearSolverConfig("scipy_splu")
    controls = model.deck.data["nonlinear"]
    method = str(controls.get("method", "newton"))
    if method not in ("newton", "modified_newton"):
        raise ModelError(f"unsupported nonlinear method {method!r}")
    line_search = str(controls.get("line_search", "none"))
    if line_search not in ("none", "backtracking"):
        raise ModelError(f"unsupported nonlinear line_search {line_search!r}")
    line_search_reduction = float(controls.get("line_search_reduction", 0.5))
    line_search_armijo = float(controls.get("line_search_armijo", 1.0e-4))
    line_search_min_alpha = float(controls.get("line_search_min_alpha", 1.0e-4))
    line_search_max_backtracks = int(controls.get("line_search_max_backtracks", 14))
    if line_search == "backtracking" and not (
        0.0 < line_search_reduction < 1.0
        and 0.0 <= line_search_armijo < 1.0
        and 0.0 < line_search_min_alpha <= 1.0
        and line_search_max_backtracks >= 0
    ):
        raise ModelError("invalid Newton backtracking parameters")
    max_iterations = int(controls["max_iterations"])
    C = constraints.C
    d = constraints.rhs(t_np1)
    f_ext = assemble_external(model, t_np1)
    u = u_n.copy()
    lambdas = lambda_n.copy()
    if lambdas.shape != (C.shape[0],):
        lambdas = np.zeros(C.shape[0])
    last_du = np.zeros_like(u)
    factor = None
    frozen_K = None
    final_assembly: AssemblyResult | None = None
    pending_assembly: AssemblyResult | None = None
    ordering = str(model.deck.data["linear_solver"].get("ordering", "COLAMD"))

    try:
        for iteration in range(max_iterations + 1):
            iteration_start = perf_counter_ns()
            refresh = method == "newton" or iteration == 0
            assembly_wall_seconds = 0.0
            assembly_reused = pending_assembly is not None
            if pending_assembly is None:
                assembly_start = perf_counter_ns()
                try:
                    assembly = assemble_internal(
                        model,
                        u_n,
                        u,
                        t_n,
                        t_np1,
                        refresh,
                        element_executor=element_executor,
                        collect_timing=debug_timing,
                    )
                except RecoverableError as exc:
                    assembly_wall_seconds = (perf_counter_ns() - assembly_start) * 1.0e-9
                    history.append(
                        {
                            "t_n": t_n,
                            "t_np1": t_np1,
                            "iteration": iteration,
                            "failure_kind": "recoverable",
                            "failure_message": str(exc),
                            "assembly_wall_seconds": assembly_wall_seconds,
                            "line_search_assembly_wall_seconds": 0.0,
                            "kkt_factorization_wall_seconds": 0.0,
                            "kkt_solve_wall_seconds": 0.0,
                            "iteration_wall_seconds": (perf_counter_ns() - iteration_start) * 1.0e-9,
                        }
                    )
                    raise
                assembly_wall_seconds = (perf_counter_ns() - assembly_start) * 1.0e-9
            else:
                assembly = pending_assembly
                pending_assembly = None
            final_assembly = assembly
            r_u = assembly.f_int - f_ext + C.T @ lambdas
            r_c = C @ u - d
            force_scale = max(_inf_norm(assembly.f_int), _inf_norm(f_ext), _inf_norm(C.T @ lambdas))
            constraint_scale = max(_inf_norm(C @ u), _inf_norm(d))
            force_tol = float(controls["force_atol"]) + float(controls["force_rtol"]) * force_scale
            constraint_tol = float(controls["constraint_atol"]) + float(controls["constraint_rtol"]) * constraint_scale
            displacement_tol = float(controls.get("displacement_atol", 0.0)) + float(
                controls.get("displacement_rtol", 0.0)
            ) * max(_inf_norm(u), _inf_norm(u_n))
            residual_ok = _inf_norm(r_u) <= force_tol and _inf_norm(r_c) <= constraint_tol
            displacement_ok = not controls.get("check_displacement_increment", False) or _inf_norm(last_du) <= displacement_tol
            record = {
                "t_n": t_n, "t_np1": t_np1, "iteration": iteration,
                "force_residual_inf": _inf_norm(r_u), "force_tolerance": force_tol,
                "constraint_residual_inf": _inf_norm(r_c), "constraint_tolerance": constraint_tol,
                "displacement_increment_inf": _inf_norm(last_du),
                "assembly_wall_seconds": assembly_wall_seconds,
                "assembly_reused_from_line_search": assembly_reused,
                "line_search_assembly_wall_seconds": 0.0,
                "kkt_factorization_wall_seconds": 0.0,
                "kkt_solve_wall_seconds": 0.0,
            }
            if debug_timing:
                record["element_phase_wall_seconds"] = (
                    0.0 if assembly_reused or assembly.timing is None
                    else assembly.timing.element_phase_wall_seconds
                )
                record["sparse_finalize_wall_seconds"] = (
                    0.0 if assembly_reused or assembly.timing is None
                    else assembly.timing.sparse_finalize_wall_seconds
                )
            history.append(record)
            if iteration_observer is not None:
                iteration_observer(record, assembly)
            capture_path = None
            if capture_attempt and iteration in (0, 1):
                # Save the tangent actually used: current for exact Newton,
                # frozen iteration-zero K for modified Newton.
                capture_K = assembly.K if refresh else frozen_K
                assert capture_K is not None
                capture_path = _capture_kkt_system(
                    model, capture_K, C, r_u, r_c, t_n, t_np1, iteration,
                )
            try:
                if residual_ok and displacement_ok:
                    return _NewtonResult(u, lambdas, assembly, iteration)
                if iteration == max_iterations:
                    break
                if refresh:
                    assert assembly.K is not None
                    frozen_K = assembly.K
                    if linear_system_observer is not None:
                        linear_system_observer(frozen_K)
                    with timer(record, "kkt_factorization_wall_seconds"):
                        factor = _factor_kkt(frozen_K, C, ordering, linear_solver)
                assert factor is not None
                try:
                    with timer(record, "kkt_solve_wall_seconds"):
                        correction = solve_factor(
                            factor,
                            -np.concatenate([np.asarray(r_u), np.asarray(r_c)]),
                            linear_solver,
                        )
                finally:
                    if linear_solver.backend == "petsc_lgmres":
                        record["linear_solve"] = dict(factor.diagnostics)
                if capture_path is not None:
                    # Optional in the old format; useful as a direct reference.
                    # A failed solve leaves K/C/residuals available without it.
                    with h5py.File(capture_path, "r+") as archive:
                        archive[f"t_n={t_n}/iteration={iteration}"].create_dataset(
                            "correction", data=correction,
                        )
                if method == "newton":
                    close_factor(factor)
                    factor = None
                if not np.all(np.isfinite(correction)):
                    raise ModelError("KKT solve returned a non-finite correction")
                delta_u = correction[:model.mesh.ndof]
                delta_lambda = correction[model.mesh.ndof:]
                if line_search == "none":
                    last_du = delta_u
                    u += delta_u
                    lambdas += delta_lambda
                    continue

                # Trial every candidate from the same committed material state.  Invalid
                # intermediate configurations reject only the candidate, not the increment.
                base_merit = _scaled_residual_merit(r_u, r_c, force_tol, constraint_tol)
                alpha = 1.0
                candidate_failures: list[str] = []
                accepted = False
                attempted_candidates = 0
                line_search_element_seconds = 0.0
                line_search_sparse_seconds = 0.0
                for backtrack in range(line_search_max_backtracks + 1):
                    if alpha < line_search_min_alpha:
                        break
                    attempted_candidates += 1
                    candidate_u = u + alpha * delta_u
                    candidate_lambdas = lambdas + alpha * delta_lambda
                    candidate_start = perf_counter_ns()
                    try:
                        candidate = assemble_internal(
                            model,
                            u_n,
                            candidate_u,
                            t_n,
                            t_np1,
                            method == "newton",
                            element_executor=element_executor,
                            collect_timing=debug_timing,
                        )
                    except RecoverableError as exc:
                        candidate_failures.append(str(exc))
                    else:
                        if debug_timing and candidate.timing is not None:
                            line_search_element_seconds += candidate.timing.element_phase_wall_seconds
                            line_search_sparse_seconds += candidate.timing.sparse_finalize_wall_seconds
                        candidate_r_u = candidate.f_int - f_ext + C.T @ candidate_lambdas
                        candidate_r_c = C @ candidate_u - d
                        candidate_merit = _scaled_residual_merit(
                            candidate_r_u, candidate_r_c, force_tol, constraint_tol
                        )
                        if candidate_merit <= (1.0 - line_search_armijo * alpha) * base_merit:
                            u = candidate_u
                            lambdas = candidate_lambdas
                            last_du = alpha * delta_u
                            record["line_search_alpha"] = alpha
                            record["line_search_backtracks"] = backtrack
                            record["line_search_trials"] = attempted_candidates
                            record["line_search_merit"] = candidate_merit
                            record["line_search_recoverable_rejections"] = len(candidate_failures)
                            if candidate_failures:
                                record["line_search_last_failure"] = candidate_failures[-1]
                            pending_assembly = candidate
                            accepted = True
                            break
                    finally:
                        record["line_search_assembly_wall_seconds"] += (
                            perf_counter_ns() - candidate_start
                        ) * 1.0e-9
                    alpha *= line_search_reduction
                if debug_timing:
                    record["line_search_element_phase_wall_seconds"] = line_search_element_seconds
                    record["line_search_sparse_finalize_wall_seconds"] = line_search_sparse_seconds
                if not accepted:
                    record["line_search_alpha"] = None
                    record["line_search_backtracks"] = attempted_candidates
                    record["line_search_trials"] = attempted_candidates
                    record["line_search_recoverable_rejections"] = len(candidate_failures)
                    if candidate_failures:
                        record["line_search_last_failure"] = candidate_failures[-1]
                    raise RecoverableError(
                        "Newton line search could not find an admissible residual-reducing step"
                    )
            finally:
                record["iteration_wall_seconds"] = (perf_counter_ns() - iteration_start) * 1.0e-9
        assert final_assembly is not None
        raise RecoverableError(f"global Newton failed in {max_iterations} iterations")
    finally:
        close_factor(factor)


def _matches(t: float, values: set[float], tol: float) -> bool:
    return any(abs(t - value) <= tol for value in values)


def _load_run_log_prefix(
    path: Path,
    restart_time: float,
    analysis_start_time: float,
    tolerance: float,
) -> tuple[list[IncrementRecord], list[dict], float]:
    if not path.is_file():
        raise ModelError("cannot resume an existing results database without its run log")
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read run log {path}") from exc
    try:
        increments = [
            IncrementRecord(**row)
            for row in payload.get("increments", [])
            if float(row["t_np1"]) <= restart_time + tolerance
        ]
        newton = [
            row
            for row in payload.get("newton_history", [])
            if float(row["t_np1"]) <= restart_time + tolerance
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError(f"run log {path} has invalid increment history") from exc
    if restart_time > analysis_start_time + tolerance and (
        not increments or abs(increments[-1].t_np1 - restart_time) > tolerance
    ):
        raise ModelError("run log has no accepted increment at the restart time")
    elapsed_raw = payload.get("timing", {}).get("elapsed_wall_seconds", 0.0)
    elapsed = float(elapsed_raw) if elapsed_raw is not None else 0.0
    return increments, newton, elapsed


def run_analysis(
    deck_or_path: PreparedAnalysis | Deck | str | Path,
    *,
    stop_time: float | None = None,
    num_processes: int = 1,
    debug_timing: bool = False,
    solver_backend: str | None = None,
    num_threads: int | None = None,
    petsc_options: str | Path | None = None,
) -> AnalysisResult | None:
    """Run one FE analysis; MPI ranks other than zero serve PETSc solves only."""
    analysis_wall_start = perf_counter_ns()
    source = (
        load_deck(deck_or_path)
        if isinstance(deck_or_path, (str, Path)) else deck_or_path
    )
    deck = source.deck if isinstance(source, PreparedAnalysis) else source
    linear_solver = LinearSolverConfig(
        solver_backend or str(deck.data["linear_solver"]["backend"]), num_threads,
    )
    options = dict(
        stop_time=stop_time, num_processes=num_processes, debug_timing=debug_timing,
        analysis_wall_start=analysis_wall_start,
    )
    if linear_solver.backend != "petsc_lgmres":
        if any(int(os.environ.get(name, "1")) > 1 for name in (
            "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE",
        )):
            raise ModelError("multi-rank MPI launch requires petsc_lgmres; run direct backends without mpiexec")
        if petsc_options is not None:
            raise ModelError("--petsc-options is valid only with petsc_lgmres")
        return _run_analysis(source, linear_solver=linear_solver, **options)

    # Lazy import and initialization occur here, never at module import or in
    # element-worker startup. Only root constructs the mesh, pool and writers.
    from .petsc_solver import PetscSession

    with PetscSession(petsc_options) as session:
        if session.rank != 0:
            status = session.serve()
            if status:
                raise SystemExit(status)
            return None
        linear_solver = LinearSolverConfig("petsc_lgmres", petsc_session=session)
        return _run_analysis(source, linear_solver=linear_solver, **options)


def _run_analysis(
    deck_or_path: PreparedAnalysis | Deck,
    *,
    stop_time: float | None,
    num_processes: int,
    debug_timing: bool,
    linear_solver: LinearSolverConfig,
    analysis_wall_start: int | None = None,
) -> AnalysisResult:
    if analysis_wall_start is None:
        analysis_wall_start = perf_counter_ns()
    prepared = deck_or_path if isinstance(deck_or_path, PreparedAnalysis) else prepare_analysis(deck_or_path)
    deck = prepared.deck
    mesh = prepared.mesh
    model = prepared.model
    constraints = prepared.constraints
    events = prepared.events.copy()
    restart_times = prepared.restart_times
    analysis = deck.data["analysis"]
    t_start, t_end = float(analysis["t_start"]), float(analysis["t_end"])
    if stop_time is not None:
        if not t_start < stop_time <= t_end:
            raise ModelError("stop_time must lie inside the analysis interval")
        t_end = float(stop_time)
        events = np.unique(np.append(events[events <= t_end], t_end))
    time_data = deck.data["time"]
    proposed_dt = float(time_data["dt_initial"])
    u_n = np.zeros(mesh.ndof)
    lambda_n = np.zeros(constraints.C.shape[0])
    t_n = t_start
    restart_from = str(deck.data["restart"].get("restart_from", ""))
    if restart_from:
        # Restore constitutive history and accepted stress for reporting in place.
        # The next positive-time trial supplies fresh stress and its tangent;
        # cached stress is not passed to constitutive evolution.
        t_n, proposed_dt, u_n, lambda_n = load_restart(deck.resolve(restart_from), model)
        if lambda_n.shape != (constraints.C.shape[0],):
            raise ModelError("restart multiplier shape mismatch")
    tol_time = 1.0e-12 * max(1.0, abs(t_start), abs(t_end))
    if t_n > t_end + tol_time:
        raise ModelError("restart time is beyond the requested analysis endpoint")

    output_dir = deck.resolve(str(deck.data["output"]["directory"]))
    database_path = output_dir / str(deck.data["output"].get("database", "run.h5"))
    log_path = output_dir / str(deck.data["output"].get("history_file", "run_log.json"))
    prior_increments: list[IncrementRecord] = []
    prior_newton: list[dict] = []
    elapsed_before_restart = 0.0
    resuming_existing_database = bool(restart_from) and database_path.is_file()
    if resuming_existing_database:
        prior_increments, prior_newton, elapsed_before_restart = _load_run_log_prefix(
            log_path, t_n, t_start, tol_time
        )
    dt_min = float(time_data["dt_min"])
    dt_max = float(time_data["dt_max"])
    cutback_factor = float(time_data["cutback_factor"])
    growth_factor = float(time_data["growth_factor"])
    if not (
        0.0 < dt_min <= proposed_dt <= dt_max
        and 0.0 < cutback_factor < 1.0
        and growth_factor >= 1.0
        and int(time_data["max_attempts_per_increment"]) > 0
    ):
        raise ModelError("invalid pseudo-time controller parameters")
    restart_index = sum(value <= t_n + tol_time for value in restart_times)
    executor = ElementExecutor(
        (block.material for block in model.blocks),
        sum(len(block.connectivity) for block in model.blocks),
        num_processes,
    )
    analysis_summary = build_analysis_summary(
        model,
        analysis_start=t_start,
        analysis_end=float(analysis["t_end"]),
        run_start=t_n,
        run_target=t_end,
        restarted=bool(restart_from),
        solution_method="lagrange_multiplier_kkt",
        solution_backend=linear_solver.backend,
        unknowns_by_type={
            "displacement": int(mesh.ndof),
            "multiplier": int(constraints.C.shape[0]),
        },
        solution_options={
            **(
                {"ordering": str(deck.data["linear_solver"].get("ordering", "COLAMD"))}
                if linear_solver.backend != "petsc_lgmres"
                else {}
            ),
            **(
                {"num_threads": linear_solver.num_threads}
                if linear_solver.backend == "superlu_mt"
                else {}
            ),
            **(
                linear_solver.petsc_session.info
                if linear_solver.backend == "petsc_lgmres"
                else {}
            ),
        },
    )
    print_startup_summary(analysis_summary)
    print(executor.describe())
    result = AnalysisResult(
        model,
        constraints,
        t_n,
        u_n,
        lambda_n,
        increments=prior_increments,
        newton_history=prior_newton,
        analysis=analysis_summary,
        execution=executor.info.as_dict(),
    )

    def observe_linear_system(K: sparse.csr_matrix) -> None:
        linear = result.analysis["linear_system"]
        if "matrices" in linear:
            return
        stiffness_entries = int(K.nnz)
        kkt_entries = stiffness_entries + 2 * int(constraints.C.nnz)
        kkt_order = int(K.shape[0] + constraints.C.shape[0])
        record_sparse_system(
            result.analysis,
            primary_name="K",
            primary_shape=K.shape,
            primary_stored_entries=stiffness_entries,
            system_name="KKT",
            system_shape=(kkt_order, kkt_order),
            system_stored_entries=kkt_entries,
        )
        print_linear_system_summary(result.analysis)

    writer: HDF5ResultWriter | None = None
    try:
        writer = HDF5ResultWriter(
            database_path,
            model,
            resume=bool(restart_from),
            resume_time=t_n if resuming_existing_database else None,
        )
        result.timing["elapsed_wall_seconds"] = (
            elapsed_before_restart + (perf_counter_ns() - analysis_wall_start) * 1.0e-9
        )
        write_run_log(
            log_path,
            result.increments,
            result.newton_history,
            result.verification,
            result.execution,
            result.timing,
            result.analysis,
        )
        # Startup is observation, not a zero-duration constitutive increment.
        initial_assembly = observe_committed(model, u_n)
        writer.append(t_n, u_n, np.asarray(-constraints.C.T @ lambda_n), initial_assembly)
        final_verified = False
        while t_n < t_end - tol_time:
            future_events = events[events > t_n + tol_time]
            next_event = float(future_events[0]) if len(future_events) else t_end
            attempts = 0
            cutbacks = 0
            last_failure = None
            while True:
                attempts += 1
                if attempts > int(time_data["max_attempts_per_increment"]):
                    raise ModelError(
                        f"maximum increment attempts exceeded; last failure: {last_failure}"
                    ) from last_failure
                # Release the previous traceback/assembly before the next attempt.
                last_failure = None
                event_gap = next_event - t_n
                dt = min(proposed_dt, event_gap, t_end - t_n)
                event_landing_below_min = event_gap < dt_min + tol_time and abs(dt - event_gap) <= tol_time
                if dt < dt_min - tol_time and not event_landing_below_min:
                    raise ModelError("cutback requires a pseudo-time increment below dt_min")
                t_trial = t_n + dt
                if abs(t_trial - next_event) <= tol_time:
                    t_trial = next_event
                history_start = len(result.newton_history)
                try:
                    trial = _newton_attempt(
                        model,
                        constraints,
                        u_n,
                        lambda_n,
                        t_n,
                        t_trial,
                        result.newton_history,
                        element_executor=executor,
                        debug_timing=debug_timing,
                        linear_system_observer=observe_linear_system,
                        linear_solver=linear_solver,
                        capture_attempt=(t_n == 0.0 and attempts == 1),
                    )
                except RecoverableError as exc:
                    last_failure = exc
                    cutbacks += 1
                    proposed_dt = min(proposed_dt, dt) * cutback_factor
                    if len(result.newton_history) == history_start:
                        result.newton_history.append({"t_n": t_n, "t_np1": t_trial})
                    result.newton_history[-1].update({
                        "failure_kind": "recoverable",
                        "failure_message": str(exc),
                        "increment_attempt": attempts,
                    })
                    # Save rejected attempts before retry/termination, without
                    # committing their material state or writing field results.
                    result.timing["elapsed_wall_seconds"] = (
                        elapsed_before_restart
                        + (perf_counter_ns() - analysis_wall_start) * 1.0e-9
                    )
                    write_run_log(
                        log_path,
                        result.increments,
                        result.newton_history,
                        result.verification,
                        result.execution,
                        result.timing,
                        result.analysis,
                    )
                    if event_landing_below_min or proposed_dt < dt_min - tol_time:
                        raise ModelError(
                            "recoverable failure cannot be cut back without violating dt_min; "
                            f"last failure: {exc}"
                        ) from exc
                    continue
                if abs(t_trial - t_end) <= tol_time:
                    from .verification import verify_analysis
                    # Tangent checks must use this increment's original history,
                    # before it is overwritten by the accepted endpoint state.
                    result.verification = verify_analysis(
                        model, constraints, t_trial, trial.u, trial.lambdas,
                        assembly=trial.assembly, u_n=u_n, t_n=t_n,
                        element_executor=executor,
                    )
                    final_verified = True
                commit_trial_states(model, trial.assembly.state_trial, trial.assembly.gauss_output)
                old_t = t_n
                t_n, u_n, lambda_n = t_trial, trial.u, trial.lambdas
                result.increments.append(IncrementRecord(old_t, t_n, attempts, trial.iterations, cutbacks))
                result.t, result.u, result.lambdas = t_n, u_n, lambda_n
                reaction = np.asarray(-constraints.C.T @ lambda_n)
                writer.append(t_n, u_n, reaction, trial.assembly)
                if trial.iterations <= int(time_data["grow_if_newton_iterations_le"]):
                    proposed_dt = min(proposed_dt * growth_factor, dt_max)
                else:
                    proposed_dt = min(proposed_dt, dt_max)
                if _matches(t_n, restart_times, tol_time):
                    restart_index += 1
                    restart_dir = output_dir / "restart"
                    write_restart(
                        restart_dir / f"restart_{restart_index:06d}.h5",
                        model, t_n, proposed_dt, u_n, lambda_n,
                    )
                    keep_last = int(deck.data["restart"].get("keep_last", 0))
                    if keep_last > 0:
                        files = sorted(restart_dir.glob("restart_*.h5"))
                        for obsolete in files[:-keep_last]:
                            obsolete.unlink()
                result.timing["elapsed_wall_seconds"] = (
                    elapsed_before_restart
                    + (perf_counter_ns() - analysis_wall_start) * 1.0e-9
                )
                write_run_log(
                    log_path,
                    result.increments,
                    result.newton_history,
                    result.verification,
                    result.execution,
                    result.timing,
                    result.analysis,
                )
                break
        from .verification import verify_analysis
        if not final_verified:
            result.verification = verify_analysis(
                model, constraints, result.t, result.u, result.lambdas,
                assembly=initial_assembly, element_executor=executor,
            )
        result.timing["elapsed_wall_seconds"] = (
            elapsed_before_restart + (perf_counter_ns() - analysis_wall_start) * 1.0e-9
        )
        write_run_log(
            log_path,
            result.increments,
            result.newton_history,
            result.verification,
            result.execution,
            result.timing,
            result.analysis,
        )
        return result
    finally:
        if writer is not None:
            writer.close()
        executor.close()
