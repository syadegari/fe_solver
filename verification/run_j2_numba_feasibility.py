"""Benchmark equivalent interpreted and Numba J2 material kernels."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

from fe_solver.config import Deck, load_deck
from fe_solver.j2_kernel import (
    available_j2_backends,
    configure_j2_backend,
    numba_signatures,
)
from fe_solver.materials import (
    init_j2_plasticity,
    j2_plasticity_definition,
    update_j2_plasticity,
)
from fe_solver.solver import run_analysis
from fe_solver.types import MaterialInitRequest, MaterialRequest, ModelError
from verification.compare_run_databases import compare_databases


ROOT = Path(__file__).resolve().parents[1]
PRISM_DECK = ROOT / "examples/j2_necking_prism_small_hex8_fbar.toml"
DEFAULT_RESULT_ROOT = ROOT / "examples/results/j2_numba_feasibility"
DEFAULT_MATERIAL_REPORT = ROOT / "verification/j2_numba_results/material_benchmark.json"


def _properties() -> dict[str, float]:
    return {
        "shear_modulus": 80193.8,
        "bulk_modulus": 164210.0,
        "initial_yield_stress": 450.0,
        "linear_hardening_modulus": 129.24,
        "saturation_increment": 265.0,
        "saturation_rate": 16.93,
    }


def _initial_request(F: np.ndarray, need_tangent: bool) -> MaterialRequest:
    material = j2_plasticity_definition("benchmark_steel", _properties())
    initialized = init_j2_plasticity(
        MaterialInitRequest(material.properties, None, np.zeros(3), 0.0)
    )
    if not initialized.status.ok:
        raise ModelError(initialized.status.message)
    return MaterialRequest(
        np.eye(3), F, material.state_layout.view(initialized.state0),
        material.properties, None, 0.0, 1.0, need_tangent,
    )


def _relative_max(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), 1.0)
    return float(np.max(np.abs(candidate - reference) / scale))


def _evaluate(backend: str, request: MaterialRequest):
    configure_j2_backend(backend)
    response = update_j2_plasticity(request)
    if not response.status.ok:
        raise ModelError(response.status.message)
    return response


def _fixed_case_equivalence(F: np.ndarray, need_tangent: bool) -> dict[str, float]:
    request = _initial_request(F, need_tangent)
    python = _evaluate("python", request)
    numba = _evaluate("numba", request)
    report = {
        "P_max_absolute": float(np.max(np.abs(numba.P - python.P))),
        "P_max_relative": _relative_max(python.P, numba.P),
        "state_max_absolute": float(
            np.max(np.abs(numba.state_trial.values - python.state_trial.values))
        ),
        "state_max_relative": _relative_max(
            python.state_trial.values, numba.state_trial.values
        ),
    }
    if need_tangent:
        assert python.A_alg is not None and numba.A_alg is not None
        report.update(
            A_max_absolute=float(np.max(np.abs(numba.A_alg - python.A_alg))),
            A_max_relative=_relative_max(python.A_alg, numba.A_alg),
        )
    return report


def _path_equivalence(number_of_steps: int) -> dict[str, float]:
    material = j2_plasticity_definition("benchmark_steel", _properties())
    initialized = init_j2_plasticity(
        MaterialInitRequest(material.properties, None, np.zeros(3), 0.0)
    )
    states: dict[str, np.ndarray] = {}
    stresses: dict[str, np.ndarray] = {}
    tangents: dict[str, np.ndarray] = {}
    for backend in ("python", "numba"):
        configure_j2_backend(backend)
        state_values = initialized.state0.copy()
        F_n = np.eye(3)
        P = np.zeros((3, 3))
        A = np.zeros((3, 3, 3, 3))
        for index in range(1, number_of_steps + 1):
            strain = 0.1 * index / number_of_steps
            F = np.diag(
                [np.exp(strain), np.exp(-0.5 * strain), np.exp(-0.5 * strain)]
            )
            response = update_j2_plasticity(
                MaterialRequest(
                    F_n, F, material.state_layout.view(state_values),
                    material.properties, None, (index - 1) / number_of_steps,
                    index / number_of_steps, True,
                )
            )
            if not response.status.ok:
                raise ModelError(response.status.message)
            state_values = response.state_trial.values.copy()
            F_n = F
            P = response.P
            assert response.A_alg is not None
            A = response.A_alg
        states[backend] = state_values
        stresses[backend] = P
        tangents[backend] = A
    return {
        "steps": number_of_steps,
        "P_max_absolute": float(np.max(np.abs(stresses["numba"] - stresses["python"]))),
        "P_max_relative": _relative_max(stresses["python"], stresses["numba"]),
        "state_max_absolute": float(np.max(np.abs(states["numba"] - states["python"]))),
        "state_max_relative": _relative_max(states["python"], states["numba"]),
        "A_max_absolute": float(np.max(np.abs(tangents["numba"] - tangents["python"]))),
        "A_max_relative": _relative_max(tangents["python"], tangents["numba"]),
    }


def _time_request(
    backend: str, request: MaterialRequest, calls: int, repeats: int
) -> dict[str, object]:
    configure_j2_backend(backend)
    first_start = perf_counter_ns()
    first = update_j2_plasticity(request)
    first_seconds = (perf_counter_ns() - first_start) * 1.0e-9
    if not first.status.ok:
        raise ModelError(first.status.message)
    samples = []
    checksum = 0.0
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(calls):
            response = update_j2_plasticity(request)
            checksum += float(response.P[0, 0]) + float(response.state_trial.values[6])
        samples.append((perf_counter_ns() - start) * 1.0e-9 / calls)
    return {
        "calls_per_repeat": calls,
        "repeats": repeats,
        "first_call_seconds": first_seconds,
        "seconds_per_call": samples,
        "median_seconds_per_call": statistics.median(samples),
        "minimum_seconds_per_call": min(samples),
        "checksum": checksum,
    }


def run_material_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if "numba" not in available_j2_backends():
        raise ModelError("Numba is not installed; install the j2-jit optional dependency")
    elastic = np.diag([np.exp(0.001), np.exp(-0.0005), np.exp(-0.0005)])
    plastic = np.array(
        [[1.025, 0.012, 0.0], [0.0, 0.988, 0.004], [0.0, 0.0, 0.989]]
    )
    correctness = {}
    timing = {}
    with threadpool_limits(limits=1):
        for name, F in (("elastic", elastic), ("plastic", plastic)):
            for need_tangent in (False, True):
                label = f"{name}_{'tangent' if need_tangent else 'stress_state'}"
                correctness[label] = _fixed_case_equivalence(F, need_tangent)
                request = _initial_request(F, need_tangent)
                calls = args.tangent_calls if need_tangent else args.stress_calls
                timing[label] = {
                    backend: _time_request(backend, request, calls, args.repeats)
                    for backend in ("python", "numba")
                }
                timing[label]["warm_speedup"] = (
                    timing[label]["python"]["median_seconds_per_call"]
                    / timing[label]["numba"]["median_seconds_per_call"]
                )
        path = _path_equivalence(args.path_steps)

    try:
        import numba
        numba_version = numba.__version__
    except ImportError:  # pragma: no cover
        numba_version = None
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "numba": numba_version,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "blas_threads": 1,
        },
        "kernel": {
            "nopython_signatures": [str(value) for value in numba_signatures()],
            "fastmath": False,
        },
        "correctness": correctness,
        "sequential_uniaxial_path_correctness": path,
        "timing": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")
    return report


def run_prism(args: argparse.Namespace) -> None:
    source = load_deck(PRISM_DECK)
    data = copy.deepcopy(source.data)
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else DEFAULT_RESULT_ROOT / f"prism_{args.backend}"
    )
    data["output"]["directory"] = str(output)
    data["restart"]["restart_from"] = (
        str(args.restart_from.resolve()) if args.restart_from is not None else ""
    )
    result = run_analysis(
        Deck(source.path, data, source.curves),
        stop_time=args.stop_time,
        num_processes=args.num_processes,
        debug_timing=True,
        j2_backend=args.backend,
    )
    print(
        f"{args.backend} converged {len(result.increments)} increments to "
        f"t={result.t:.12g}; max|u|={abs(result.u).max():.6g}; "
        f"force balance={result.verification['force_balance_inf']:.3e}; "
        f"output={output}"
    )


def compare_prisms(args: argparse.Namespace) -> None:
    comparison = compare_databases(
        args.python_database, args.numba_database, rtol=args.rtol, atol=args.atol
    )
    with args.python_log.open(encoding="utf-8") as stream:
        python_log = json.load(stream)
    with args.numba_log.open(encoding="utf-8") as stream:
        numba_log = json.load(stream)
    fields = (
        "t_n", "t_np1", "iteration", "line_search_alpha",
        "line_search_backtracks", "line_search_trials",
        "line_search_recoverable_rejections",
    )
    python_path = [tuple(row.get(name) for name in fields) for row in python_log["newton_history"]]
    numba_path = [tuple(row.get(name) for name in fields) for row in numba_log["newton_history"]]
    python_seconds = float(python_log["timing"]["elapsed_wall_seconds"])
    numba_seconds = float(numba_log["timing"]["elapsed_wall_seconds"])
    report = {
        "database_comparison": comparison.as_dict(),
        "same_increment_history": python_log["increments"] == numba_log["increments"],
        "same_newton_and_line_search_path": python_path == numba_path,
        "python_elapsed_wall_seconds": python_seconds,
        "numba_elapsed_wall_seconds": numba_seconds,
        "warm_solver_speedup": python_seconds / numba_seconds,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not comparison.passed:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    material = commands.add_parser("material", help="benchmark synthetic material points")
    material.add_argument("--stress-calls", type=int, default=2000)
    material.add_argument("--tangent-calls", type=int, default=500)
    material.add_argument("--repeats", type=int, default=7)
    material.add_argument("--path-steps", type=int, default=400)
    material.add_argument("--output", type=Path, default=DEFAULT_MATERIAL_REPORT)
    material.set_defaults(function=run_material_benchmark)

    prism = commands.add_parser("prism", help="run one isolated small-prism backend")
    prism.add_argument("--backend", choices=("python", "numba"), required=True)
    prism.add_argument("--stop-time", type=float, default=0.5)
    prism.add_argument("--restart-from", type=Path)
    prism.add_argument("--num-processes", type=int, default=2)
    prism.add_argument("--output-directory", type=Path)
    prism.set_defaults(function=run_prism)

    compare = commands.add_parser("compare-prisms", help="compare isolated prism runs")
    compare.add_argument("python_database", type=Path)
    compare.add_argument("numba_database", type=Path)
    compare.add_argument("python_log", type=Path)
    compare.add_argument("numba_log", type=Path)
    compare.add_argument("--rtol", type=float, default=5.0e-12)
    compare.add_argument("--atol", type=float, default=2.0e-9)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(function=compare_prisms)

    args = parser.parse_args(argv)
    if any(
        getattr(args, name, 1) < 1
        for name in ("stress_calls", "tangent_calls", "repeats", "path_steps")
        if hasattr(args, name)
    ):
        parser.error("material benchmark counts must be positive")
    try:
        args.function(args)
    except ModelError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
