from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from superlu_mt_backend import SuperLUMTFactor, load_superlu_mt


@dataclass(frozen=True)
class SolverConfig:
    ordering: str
    diag_pivot_thresh: float
    panel_size: int
    relax: int


@dataclass(frozen=True)
class SolveResult:
    x: np.ndarray
    factor_wall_seconds: float
    solve_wall_seconds: float
    setup_seconds: float | None = None
    factor_kernel_seconds: float | None = None
    solve_kernel_seconds: float | None = None


@dataclass(frozen=True)
class ValidationMetrics:
    residual_inf: float
    backward_error_inf: float
    x_difference_inf: float | None = None
    x_relative_difference_inf: float | None = None


def _read_csr(group: h5py.Group, name: str) -> sparse.csr_matrix:
    g = group[name]
    shape = tuple(int(value) for value in g.attrs["shape"])

    # The benchmark intentionally preserves the stored representation instead
    # of silently converting it. The FE solver writes float64/int32 CSR data.
    data = np.asarray(g["data"])
    indices = np.asarray(g["indices"])
    indptr = np.asarray(g["indptr"])

    matrix = sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)

    if matrix.data.dtype != np.float64:
        raise TypeError(f"{name}.data must be float64, got {matrix.data.dtype}")
    if matrix.indices.dtype != np.int32:
        raise TypeError(f"{name}.indices must be int32, got {matrix.indices.dtype}")
    if matrix.indptr.dtype != np.int32:
        raise TypeError(f"{name}.indptr must be int32, got {matrix.indptr.dtype}")
    if not matrix.has_canonical_format:
        raise ValueError(f"stored {name} matrix is not canonical CSR")

    return matrix


def load_problem(path: str | Path, group_name: str) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Load one recorded Newton linear system and reconstruct its KKT matrix."""
    with h5py.File(path, "r") as h5:
        group = h5[group_name]
        K = _read_csr(group, "K")
        C = _read_csr(group, "C")
        r_u = np.asarray(group["r_u"])
        r_c = np.asarray(group["r_c"])

    if r_u.dtype != np.float64 or r_c.dtype != np.float64:
        raise TypeError("stored residual vectors must be float64")
    if r_u.ndim != 1 or r_c.ndim != 1:
        raise ValueError("stored residual vectors must be one-dimensional")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")
    if C.shape[1] != K.shape[0]:
        raise ValueError(f"C shape {C.shape} is incompatible with K shape {K.shape}")
    if r_u.shape != (K.shape[0],):
        raise ValueError(f"r_u shape {r_u.shape} is incompatible with K shape {K.shape}")
    if r_c.shape != (C.shape[0],):
        raise ValueError(f"r_c shape {r_c.shape} is incompatible with C shape {C.shape}")

    # Match fe_solver.solver._factor_kkt(): K and C are CSR and the KKT matrix
    # is constructed directly as CSC. No duplicate summation, sorting, dtype
    # conversion, or extra CSC copy is performed here.
    zero = sparse.csr_matrix((C.shape[0], C.shape[0]), dtype=np.float64)
    A = sparse.bmat([[K, C.T], [C, zero]], format="csc")
    b = -np.concatenate((r_u, r_c))

    if A.data.dtype != np.float64:
        raise TypeError(f"KKT data must be float64, got {A.data.dtype}")
    if A.indices.dtype != np.int32 or A.indptr.dtype != np.int32:
        raise TypeError(
            "KKT indices and indptr must be int32 for the current SuperLU_MT build"
        )
    if not A.has_canonical_format:
        raise ValueError("KKT construction did not produce canonical CSC")
    if not (A.data.flags.c_contiguous and A.indices.flags.c_contiguous and A.indptr.flags.c_contiguous):
        raise ValueError("KKT CSC arrays are not C-contiguous")
    if b.dtype != np.float64 or not b.flags.c_contiguous:
        raise ValueError("KKT RHS is not contiguous float64")

    return A, b


def solve_scipy(A: sparse.csc_matrix, b: np.ndarray, config: SolverConfig) -> SolveResult:
    start = perf_counter()
    factor = splu(
        A,
        permc_spec=config.ordering,
        diag_pivot_thresh=config.diag_pivot_thresh,
        panel_size=config.panel_size,
        relax=config.relax,
    )
    factor_wall = perf_counter() - start

    start = perf_counter()
    x = factor.solve(b)
    solve_wall = perf_counter() - start

    return SolveResult(
        x=x,
        factor_wall_seconds=factor_wall,
        solve_wall_seconds=solve_wall,
    )


def solve_superlu_mt(
    lib,
    A: sparse.csc_matrix,
    b: np.ndarray,
    config: SolverConfig,
    nprocs: int,
) -> SolveResult:
    start = perf_counter()
    factor = SuperLUMTFactor.factor(
        lib,
        A,
        nprocs=nprocs,
        ordering=config.ordering,
        diag_pivot_thresh=config.diag_pivot_thresh,
        panel_size=config.panel_size,
        relax=config.relax,
    )
    factor_wall = perf_counter() - start

    try:
        start = perf_counter()
        x = factor.solve(b)
        solve_wall = perf_counter() - start

        return SolveResult(
            x=x,
            factor_wall_seconds=factor_wall,
            solve_wall_seconds=solve_wall,
            setup_seconds=factor.setup_seconds,
            factor_kernel_seconds=factor.factor_seconds,
            solve_kernel_seconds=factor.solve_seconds,
        )
    finally:
        factor.close()


def matrix_inf_norm(A: sparse.csc_matrix) -> float:
    row_sums = np.asarray(abs(A).sum(axis=1)).ravel()
    return float(row_sums.max()) if row_sums.size else 0.0


def validation_metrics(
    A: sparse.csc_matrix,
    b: np.ndarray,
    x: np.ndarray,
    A_inf: float,
    reference: np.ndarray | None = None,
) -> ValidationMetrics:
    residual_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))
    x_inf = float(np.linalg.norm(x, ord=np.inf))
    b_inf = float(np.linalg.norm(b, ord=np.inf))
    denominator = A_inf * x_inf + b_inf
    backward_error = residual_inf / denominator if denominator else residual_inf

    if reference is None:
        return ValidationMetrics(
            residual_inf=residual_inf,
            backward_error_inf=backward_error,
        )

    difference = float(np.linalg.norm(x - reference, ord=np.inf))
    reference_inf = float(np.linalg.norm(reference, ord=np.inf))
    relative_difference = difference / max(reference_inf, np.finfo(np.float64).tiny)
    return ValidationMetrics(
        residual_inf=residual_inf,
        backward_error_inf=backward_error,
        x_difference_inf=difference,
        x_relative_difference_inf=relative_difference,
    )


def _run_repeated(
    solve: Callable[[], SolveResult],
    *,
    warmups: int,
    repeats: int,
) -> list[SolveResult]:
    for _ in range(warmups):
        solve()
    return [solve() for _ in range(repeats)]


def _seconds(values: list[float]) -> str:
    return f"median={median(values):.6f}  min={min(values):.6f}  max={max(values):.6f}"


def print_problem_summary(A: sparse.csc_matrix, b: np.ndarray, config: SolverConfig) -> None:
    print("Problem")
    print(f"  shape              : {A.shape}")
    print(f"  nnz                : {A.nnz}")
    print(f"  density            : {A.nnz / (A.shape[0] * A.shape[1]):.6e}")
    print(f"  canonical CSC      : {A.has_canonical_format}")
    print(f"  data/index dtype   : {A.data.dtype} / {A.indices.dtype}")
    print(f"  RHS shape/dtype    : {b.shape} / {b.dtype}")
    print()
    print("Common solver parameters")
    print(f"  ordering           : {config.ordering}")
    print(f"  diag_pivot_thresh  : {config.diag_pivot_thresh:g}")
    print(f"  panel_size         : {config.panel_size}")
    print(f"  relax              : {config.relax}")
    print()

    thread_env = ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]
    configured = [f"{name}={os.environ[name]}" for name in thread_env if name in os.environ]
    print("BLAS/OpenMP environment")
    print(f"  {', '.join(configured) if configured else 'not explicitly constrained'}")
    print()


def print_benchmark_summary(label: str, runs: list[SolveResult]) -> None:
    print(label)
    print(f"  factor wall        : {_seconds([run.factor_wall_seconds for run in runs])} s")
    print(f"  solve wall         : {_seconds([run.solve_wall_seconds for run in runs])} s")

    if runs[0].setup_seconds is not None:
        print(f"  MT setup kernel    : {_seconds([run.setup_seconds for run in runs if run.setup_seconds is not None])} s")
        print(f"  MT pdgstrf kernel  : {_seconds([run.factor_kernel_seconds for run in runs if run.factor_kernel_seconds is not None])} s")
        print(f"  MT dgstrs kernel   : {_seconds([run.solve_kernel_seconds for run in runs if run.solve_kernel_seconds is not None])} s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark the SuperLU_MT bridge against scipy.sparse.linalg.splu"
    )
    parser.add_argument("h5", type=Path, help="HDF5 file containing a recorded FE linear system")
    parser.add_argument("group", help="HDF5 group, for example 't_n=0.02/iteration=2'")
    parser.add_argument(
        "--lib",
        type=Path,
        default=Path("build/native/libsuperlu_mt_bridge.so"),
        help="path to libsuperlu_mt_bridge.so",
    )
    parser.add_argument(
        "--ordering",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default="COLAMD",
    )
    parser.add_argument("--diag-pivot-thresh", type=float, default=1.0)
    parser.add_argument(
        "--panel-size",
        type=int,
        default=20,
        help="explicit common panel size for both backends (default: 20)",
    )
    parser.add_argument(
        "--relax",
        type=int,
        default=6,
        help="explicit common relax value for both backends (default: 6)",
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1.0e-9, help="solution comparison relative tolerance")
    parser.add_argument("--atol", type=float, default=1.0e-11, help="solution comparison absolute tolerance")
    args = parser.parse_args()

    if args.panel_size <= 0:
        parser.error("--panel-size must be positive; benchmark parameters are explicit for both backends")
    if args.relax < 0:
        parser.error("--relax must be non-negative; benchmark parameters are explicit for both backends")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.threads or any(value < 1 for value in args.threads):
        parser.error("--threads values must all be positive")
    if len(set(args.threads)) != len(args.threads):
        parser.error("--threads must not contain duplicates")

    config = SolverConfig(
        ordering=args.ordering,
        diag_pivot_thresh=args.diag_pivot_thresh,
        panel_size=args.panel_size,
        relax=args.relax,
    )

    A, b = load_problem(args.h5, args.group)
    lib = load_superlu_mt(args.lib.resolve())
    A_inf = matrix_inf_norm(A)

    print_problem_summary(A, b, config)
    print(f"Running {args.warmups} warmup(s) and {args.repeats} measured run(s) per backend configuration...")
    print()

    scipy_runs = _run_repeated(
        lambda: solve_scipy(A, b, config),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    x_reference = scipy_runs[0].x

    mt_runs_by_threads: dict[int, list[SolveResult]] = {}
    for nprocs in args.threads:
        mt_runs_by_threads[nprocs] = _run_repeated(
            lambda nprocs=nprocs: solve_superlu_mt(lib, A, b, config, nprocs),
            warmups=args.warmups,
            repeats=args.repeats,
        )

    # Validate only after all timed runs are complete, so sparse residual
    # calculations and solution comparisons cannot perturb benchmark timings.
    scipy_metrics = validation_metrics(A, b, x_reference, A_inf)
    print("Validation")
    print("  SciPy / SuperLU reference")
    print(f"    residual_inf       : {scipy_metrics.residual_inf:.6e}")
    print(f"    backward_error_inf : {scipy_metrics.backward_error_inf:.6e}")

    validation_failed = False
    for nprocs, runs in mt_runs_by_threads.items():
        metrics = [
            validation_metrics(A, b, run.x, A_inf, x_reference)
            for run in runs
        ]
        close = [
            np.allclose(run.x, x_reference, rtol=args.rtol, atol=args.atol)
            for run in runs
        ]
        validation_failed |= not all(close)

        print(f"  SuperLU_MT, {nprocs} thread(s)")
        print(f"    max residual_inf       : {max(item.residual_inf for item in metrics):.6e}")
        print(
            f"    max backward_error_inf : "
            f"{max(item.backward_error_inf for item in metrics):.6e}"
        )
        print(
            f"    max vs scipy x_inf     : "
            f"{max(item.x_difference_inf for item in metrics if item.x_difference_inf is not None):.6e}"
        )
        print(
            f"    max relative x_inf     : "
            f"{max(item.x_relative_difference_inf for item in metrics if item.x_relative_difference_inf is not None):.6e}"
        )
        print(
            f"    all measured allclose  : {all(close)}  "
            f"(rtol={args.rtol:g}, atol={args.atol:g})"
        )

    print()
    print("Benchmark")
    print_benchmark_summary("  SciPy / SuperLU", scipy_runs)

    scipy_factor_median = median(run.factor_wall_seconds for run in scipy_runs)
    for nprocs, mt_runs in mt_runs_by_threads.items():
        print_benchmark_summary(f"  SuperLU_MT, {nprocs} thread(s)", mt_runs)
        mt_factor_median = median(run.factor_wall_seconds for run in mt_runs)
        speedup = scipy_factor_median / mt_factor_median
        print(f"  factor speedup vs SciPy median: {speedup:.3f}x")

    if validation_failed:
        raise AssertionError(
            "one or more SuperLU_MT measured solutions did not match SciPy "
            f"within rtol={args.rtol:g}, atol={args.atol:g}"
        )


if __name__ == "__main__":
    main()
