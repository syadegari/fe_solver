from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, splu

from .assembly import (
    AssemblyResult,
    FEModel,
    assemble_external,
    assemble_internal,
    commit_trial_states,
)
from .config import Deck
from .constraints import ConstraintSystem
from .io import HDF5ResultWriter, load_restart, write_restart, write_run_log
from .preprocess import PreparedAnalysis, prepare_analysis
from .types import ModelError, RecoverableError


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


def _factor_kkt(K: sparse.csr_matrix, C: sparse.csr_matrix, ordering: str):
    zero = sparse.csr_matrix((C.shape[0], C.shape[0]))
    kkt = sparse.bmat([[K, C.T], [C, zero]], format="csc")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            return splu(kkt, permc_spec=ordering)
    except (RuntimeError, MatrixRankWarning) as exc:
        raise ModelError("singular sparse KKT system; check constraints and rigid modes") from exc


def _newton_attempt(
    model: FEModel,
    constraints: ConstraintSystem,
    u_n: np.ndarray,
    lambda_n: np.ndarray,
    t_n: float,
    t_np1: float,
    history: list[dict],
    iteration_observer: Callable[[dict, AssemblyResult], None] | None = None,
) -> _NewtonResult:
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

    for iteration in range(max_iterations + 1):
        refresh = method == "newton" or iteration == 0
        if pending_assembly is None:
            try:
                assembly = assemble_internal(model, u_n, u, t_n, t_np1, refresh)
            except RecoverableError as exc:
                history.append(
                    {
                        "t_n": t_n,
                        "t_np1": t_np1,
                        "iteration": iteration,
                        "failure_kind": "recoverable",
                        "failure_message": str(exc),
                    }
                )
                raise
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
        }
        history.append(record)
        if iteration_observer is not None:
            iteration_observer(record, assembly)
        if residual_ok and displacement_ok:
            return _NewtonResult(u, lambdas, assembly, iteration)
        if iteration == max_iterations:
            break
        if refresh:
            assert assembly.K is not None
            frozen_K = assembly.K
            factor = _factor_kkt(frozen_K, C, ordering)
        assert factor is not None
        correction = factor.solve(-np.concatenate([np.asarray(r_u), np.asarray(r_c)]))
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
        for backtrack in range(line_search_max_backtracks + 1):
            if alpha < line_search_min_alpha:
                break
            attempted_candidates += 1
            candidate_u = u + alpha * delta_u
            candidate_lambdas = lambdas + alpha * delta_lambda
            try:
                candidate = assemble_internal(
                    model, u_n, candidate_u, t_n, t_np1, method == "newton"
                )
            except RecoverableError as exc:
                candidate_failures.append(str(exc))
            else:
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
            alpha *= line_search_reduction
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
    assert final_assembly is not None
    raise RecoverableError(f"global Newton failed in {max_iterations} iterations")


def _matches(t: float, values: set[float], tol: float) -> bool:
    return any(abs(t - value) <= tol for value in values)


def run_analysis(
    deck_or_path: PreparedAnalysis | Deck | str | Path, *, stop_time: float | None = None
) -> AnalysisResult:
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
        t_n, proposed_dt, u_n, lambda_n = load_restart(deck.resolve(restart_from), model)
        if lambda_n.shape != (constraints.C.shape[0],):
            raise ModelError("restart multiplier shape mismatch")
    tol_time = 1.0e-12 * max(1.0, abs(t_start), abs(t_end))
    if t_n > t_end + tol_time:
        raise ModelError("restart time is beyond the requested analysis endpoint")

    result = AnalysisResult(model, constraints, t_n, u_n, lambda_n)
    output_dir = deck.resolve(str(deck.data["output"]["directory"]))
    database_path = output_dir / str(deck.data["output"].get("database", "run.h5"))
    log_path = output_dir / str(deck.data["output"].get("history_file", "run_log.json"))
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
    writer = HDF5ResultWriter(database_path, model, resume=bool(restart_from))
    try:
        initial_assembly = assemble_internal(model, u_n, u_n, t_n, t_n, False)
        writer.append(t_n, u_n, np.asarray(-constraints.C.T @ lambda_n), initial_assembly)
        while t_n < t_end - tol_time:
            future_events = events[events > t_n + tol_time]
            next_event = float(future_events[0]) if len(future_events) else t_end
            attempts = 0
            cutbacks = 0
            while True:
                attempts += 1
                if attempts > int(time_data["max_attempts_per_increment"]):
                    raise ModelError("maximum increment attempts exceeded")
                event_gap = next_event - t_n
                dt = min(proposed_dt, event_gap, t_end - t_n)
                event_landing_below_min = event_gap < dt_min + tol_time and abs(dt - event_gap) <= tol_time
                if dt < dt_min - tol_time and not event_landing_below_min:
                    raise ModelError("cutback requires a pseudo-time increment below dt_min")
                t_trial = t_n + dt
                if abs(t_trial - next_event) <= tol_time:
                    t_trial = next_event
                try:
                    trial = _newton_attempt(
                        model, constraints, u_n, lambda_n, t_n, t_trial, result.newton_history
                    )
                except RecoverableError:
                    cutbacks += 1
                    proposed_dt = min(proposed_dt, dt) * cutback_factor
                    if event_landing_below_min or proposed_dt < dt_min - tol_time:
                        raise ModelError("recoverable failure cannot be cut back without violating dt_min")
                    continue
                commit_trial_states(model, trial.assembly.state_trial)
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
                write_run_log(log_path, result.increments, result.newton_history, result.verification)
                break
        from .verification import verify_analysis
        result.verification = verify_analysis(model, constraints, result.t, result.u, result.lambdas)
        write_run_log(log_path, result.increments, result.newton_history, result.verification)
        return result
    finally:
        writer.close()
