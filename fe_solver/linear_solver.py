"""Sparse KKT factorization backends used by the nonlinear solver."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, splu
from threadpoolctl import threadpool_limits

from .types import ModelError


SUPERLU_MT_LIBRARY = (
    Path(__file__).resolve().parent.parent
    / "build"
    / "native"
    / "libsuperlu_mt_bridge.so"
)


@dataclass(frozen=True)
class LinearSolverConfig:
    backend: str
    num_threads: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"scipy_splu", "superlu_mt"}:
            raise ModelError(f"unsupported linear solver backend {self.backend!r}")
        if self.backend == "superlu_mt":
            if (
                isinstance(self.num_threads, bool)
                or not isinstance(self.num_threads, (int, np.integer))
                or self.num_threads < 1
            ):
                raise ModelError(
                    "superlu_mt requires an explicit positive --num-threads value"
                )
            object.__setattr__(self, "num_threads", int(self.num_threads))
        elif self.num_threads is not None:
            raise ModelError("--num-threads is valid only with superlu_mt")


@cache
def _load_superlu_mt():
    # Keep the optional native dependency out of the ordinary SciPy import path.
    from native.superlu_mt_backend import load_superlu_mt

    try:
        return load_superlu_mt(SUPERLU_MT_LIBRARY)
    except OSError as exc:
        raise ModelError(
            "cannot load SuperLU_MT bridge at "
            "build/native/libsuperlu_mt_bridge.so"
        ) from exc


def factor_kkt(
    K: sparse.csr_matrix,
    C: sparse.csr_matrix,
    ordering: str,
    config: LinearSolverConfig | None = None,
):
    """Build and factor the complete sparse KKT matrix."""
    config = config or LinearSolverConfig("scipy_splu")
    zero = sparse.csr_matrix((C.shape[0], C.shape[0]), dtype=C.dtype)
    kkt = sparse.bmat([[K, C.T], [C, zero]], format="csc")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            if config.backend == "scipy_splu":
                return splu(kkt, permc_spec=ordering)

            from native.superlu_mt_backend import SuperLUMTFactor

            assert config.num_threads is not None
            with threadpool_limits(limits=1):
                return SuperLUMTFactor.factor(
                    _load_superlu_mt(),
                    kkt,
                    nprocs=config.num_threads,
                    ordering=ordering,
                )
    except (RuntimeError, MatrixRankWarning) as exc:
        raise ModelError(
            f"singular sparse KKT system or installation failure; check constraints, rigid modes or installation path: {exc}"
        ) from exc


def solve_factor(factor, rhs: np.ndarray, config: LinearSolverConfig) -> np.ndarray:
    if config.backend == "superlu_mt":
        with threadpool_limits(limits=1):
            return factor.solve(rhs)
    return factor.solve(rhs)


def close_factor(factor) -> None:
    if factor is not None:
        close = getattr(factor, "close", None)
        if close is not None:
            close()
