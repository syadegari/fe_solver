from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from superlu_mt_backend import SuperLUMTFactor, load_superlu_mt


@dataclass(frozen=True)
class SolverOptions:
    ordering: str
    diag_pivot_thresh: float
    panel_size: int
    relax: int


@dataclass(frozen=True)
class SolveResult:
    x: np.ndarray
    factor_seconds: float
    solve_seconds: float
    setup_seconds: float | None = None
    factor_kernel_seconds: float | None = None
    solve_kernel_seconds: float | None = None


def _read_csr(group: h5py.Group, name: str) -> sparse.csr_matrix:
    g = group[name]
    shape = tuple(int(value) for value in g.attrs["shape"])
    return sparse.csr_matrix(
        (np.asarray(g["data"]), np.asarray(g["indices"]), np.asarray(g["indptr"])),
        shape=shape,
        copy=False,
    )


def load_problem(path: Path, group_name: str) -> tuple[sparse.csc_matrix, np.ndarray]:
    with h5py.File(path, "r") as h5:
        group = h5[group_name]
        K = _read_csr(group, "K")
        C = _read_csr(group, "C")
        r_u = np.asarray(group["r_u"])
        r_c = np.asarray(group["r_c"])

    # Match the FE solver: K and C are CSR; KKT is constructed directly as CSC.
    zero = sparse.csr_matrix((C.shape[0], C.shape[0]), dtype=np.float64)
    A = sparse.bmat([[K, C.T], [C, zero]], format="csc")
    b = -np.concatenate((r_u, r_c))
    return A, b


def solve_scipy(
    A: sparse.csc_matrix,
    b: np.ndarray,
    options: SolverOptions,
) -> SolveResult:
    start = perf_counter()
    factor = splu(
        A,
        permc_spec=options.ordering,
        diag_pivot_thresh=options.diag_pivot_thresh,
        panel_size=options.panel_size,
        relax=options.relax,
    )
    factor_seconds = perf_counter() - start

    start = perf_counter()
    x = factor.solve(b)
    solve_seconds = perf_counter() - start

    return SolveResult(x, factor_seconds, solve_seconds)


def solve_superlu_mt(
    lib,
    A: sparse.csc_matrix,
    b: np.ndarray,
    options: SolverOptions,
    nprocs: int,
) -> SolveResult:
    start = perf_counter()
    factor = SuperLUMTFactor.factor(
        lib,
        A,
        nprocs=nprocs,
        ordering=options.ordering,
        diag_pivot_thresh=options.diag_pivot_thresh,
        panel_size=options.panel_size,
        relax=options.relax,
    )
    factor_seconds = perf_counter() - start

    try:
        start = perf_counter()
        x = factor.solve(b)
        solve_seconds = perf_counter() - start

        return SolveResult(
            x,
            factor_seconds,
            solve_seconds,
            factor.setup_seconds,
            factor.factor_seconds,
            factor.solve_seconds,
        )
    finally:
        factor.close()


def residual_inf(A: sparse.csc_matrix, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - b, ord=np.inf))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark SuperLU_MT against SciPy SuperLU"
    )
    parser.add_argument("h5", type=Path)
    parser.add_argument("group")
    parser.add_argument(
        "--lib",
        type=Path,
        default=Path("build/native/libsuperlu_mt_bridge.so"),
    )
    parser.add_argument(
        "--ordering",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default="COLAMD",
    )
    parser.add_argument("--diag-pivot-thresh", type=float, default=1.0)
    parser.add_argument("--panel-size", type=int, default=20)
    parser.add_argument("--relax", type=int, default=6)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()

    options = SolverOptions(
        ordering=args.ordering,
        diag_pivot_thresh=args.diag_pivot_thresh,
        panel_size=args.panel_size,
        relax=args.relax,
    )

    A, b = load_problem(args.h5, args.group)
    lib = load_superlu_mt(args.lib.resolve())

    print(f"KKT: shape={A.shape}, nnz={A.nnz}")
    print(
        f"options: ordering={options.ordering}, "
        f"diag_pivot_thresh={options.diag_pivot_thresh:g}, "
        f"panel_size={options.panel_size}, relax={options.relax}"
    )
    print()

    scipy_result = solve_scipy(A, b, options)
    scipy_residual = residual_inf(A, scipy_result.x, b)

    print(
        f"{'solver':<18} {'factor [s]':>12} {'solve [s]':>12} "
        f"{'residual_inf':>14} {'|dx|_inf':>14} {'|dx|/|x|':>14} {'factor x':>9} {'match':>7}"
    )
    print(
        f"{'SciPy':<18} {scipy_result.factor_seconds:12.6f} "
        f"{scipy_result.solve_seconds:12.6f} {scipy_residual:14.6e} "
        f"{0.0:14.6e} {0.0:14.6e} {1.0:9.3f} {'-':>7}"
    )

    validation_failed = False

    norm = np.linalg.norm

    for nprocs in args.threads:
        result = solve_superlu_mt(lib, A, b, options, nprocs)
        residual = residual_inf(A, result.x, b)
        diff_abs = float(norm(result.x - scipy_result.x, ord=np.inf))
        diff_rel = float(norm(result.x - scipy_result.x, ord=2) / norm(scipy_result.x, ord=2))
        matches = np.allclose(result.x, scipy_result.x, rtol=1.0e-9, atol=1.0e-11)

        print(
            f"{f'SuperLU_MT ({nprocs})':<18} {result.factor_seconds:12.6f} "
            f"{result.solve_seconds:12.6f} {residual:14.6e} "
            f"{diff_abs:14.6e} "
            f"{diff_rel:14.6e} "
            f"{scipy_result.factor_seconds / result.factor_seconds:9.3f} "
            f"{str(matches):>7}"
        )
        print(
            f"{'  C kernels':<18} "
            f"setup={result.setup_seconds:.6f}  "
            f"pdgstrf={result.factor_kernel_seconds:.6f}  "
            f"dgstrs={result.solve_kernel_seconds:.6f}"
        )


if __name__ == "__main__":
    main()
