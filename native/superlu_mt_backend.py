# TODO: Testing should not test the result of factoring on its own. 
#       Currently we have:
#       residual = np.linalg.norm(A @ x - b, ord=np.inf)
#
#       but what we wanna test is how close we are to the single-thread scipy solution.
#       Schematically, we should have
#         solve_mt(A, b) -> x_mt
#         solve_scipy(A, b) -> x_scipy
#         assert np.close(x_mt, x_scipy)
#

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from scipy import sparse


ORDERINGS = {
    "NATURAL": 0,
    "MMD_ATA": 1,
    "MMD_AT_PLUS_A": 2,
    "COLAMD": 3,
}


class SuperLUMTConfig(ctypes.Structure):
    _fields_ = [
        ("permc_spec", ctypes.c_int),
        ("diag_pivot_thresh", ctypes.c_double),
        ("panel_size", ctypes.c_int),
        ("relax", ctypes.c_int),
    ]


_C_INT = ctypes.c_int
_C_INT_P = ctypes.POINTER(ctypes.c_int)    
_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)    


_FLOAT64_ARRAY_p = np.ctypeslib.ndpointer(
    dtype=np.float64,
    ndim=1,
    flags="C_CONTIGUOUS",
)

_INT32_ARRAY_p = np.ctypeslib.ndpointer(
    dtype=np.int32,
    ndim=1,
    flags="C_CONTIGUOUS",
)


def load_superlu_mt(path: str | Path):
    lib = ctypes.CDLL(str(Path(path)))

    # The signatures
    lib.slumt_factor.argtypes = [
        _C_INT,                       # nprocs
        _C_INT,                       # n
        _C_INT,                       # nnz
        _FLOAT64_ARRAY_p,                   # data
        _INT32_ARRAY_p,                     # indices
        _INT32_ARRAY_p,                     # indptr
        ctypes.POINTER(SuperLUMTConfig),    # config
        _C_DOUBLE_P,    # setup_seconds
        _C_DOUBLE_P,    # factor_seconds
        _C_INT_P,       # info_out
    ]
    lib.slumt_factor.restype = ctypes.c_void_p

    lib.slumt_solve.argtypes = [
        ctypes.c_void_p,
        _FLOAT64_ARRAY_p,
        _FLOAT64_ARRAY_p,
        _C_INT,
        _C_DOUBLE_P,
    ]
    lib.slumt_solve.restype = ctypes.c_int

    lib.slumt_free.argtypes = [
        ctypes.c_void_p,
    ]
    lib.slumt_free.restype = None

    return lib


class SuperLUMTFactor:
    def __init__(
        self,
        lib,
        handle,
        n: int,
        setup_seconds: float,
        factor_seconds: float,
    ):
        self._lib = lib
        self._handle = handle
        self.n = n
        self.setup_seconds = setup_seconds
        self.factor_seconds = factor_seconds
        self.solve_seconds = 0.0

    @classmethod
    def factor(
        cls,
        lib,
        A: sparse.spmatrix,
        *,
        nprocs: int,
        ordering: str = "COLAMD",
        diag_pivot_thresh: float = 1.0,
        panel_size: int = -1,
        relax: int = -1,
    ) -> "SuperLUMTFactor":
        if ordering not in ORDERINGS:
            raise ValueError(
                f"unsupported SuperLU_MT ordering {ordering!r}"
            )

        if not sparse.issparse(A) or A.format != "csc":
            raise TypeError(
                "SuperLU_MT backend requires a CSC sparse matrix"
            )

        if not A.has_canonical_format:
            raise ValueError(
                "SuperLU_MT backend requires canonical CSC "
                "(sorted indices and no duplicate entries)"
            )

        if A.shape[0] != A.shape[1]:
            raise ValueError("SuperLU_MT requires a square matrix")

        int32_max = np.iinfo(np.int32).max
        if A.shape[0] > int32_max or A.nnz > int32_max:
            raise OverflowError(
                "matrix is too large for this 32-bit SuperLU_MT build"
            )
        # Work on our own canonical CSC representation.
        # TODO change of format should be outside of the bridge and solve take care of it. 
        #      the backend resposibility is matrix -> factor obj -> solve
        #
        # for example, in solver.py we manage it like this
        # def _factor_kkt(K: sparse.csr_matrix, C: sparse.csr_matrix, ordering: str):
        #     zero = sparse.csr_matrix((C.shape[0], C.shape[0]))
        #     kkt = sparse.bmat([[K, C.T], [C, zero]], format="csc")
        #     try:
        #         with warnings.catch_warnings():
        #             warnings.simplefilter("error", MatrixRankWarning)
        #             return splu(kkt, permc_spec=ordering)
        #     except (RuntimeError, MatrixRankWarning) as exc:
        #         raise ModelError("singular sparse KKT system; check constraints and rigid modes") from exc
        A = A.tocsc(copy=True)
        # TODO what are the purpose of these two lines. If they are needed to stay, time them in debug mode
        #      and later decide wether to keep them or discard them
        A.sum_duplicates()
        A.sort_indices()

        A.data = np.ascontiguousarray(A.data, dtype=np.float64)
        A.indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        A.indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)

        config = SuperLUMTConfig(
            permc_spec=ORDERINGS[ordering],
            diag_pivot_thresh=diag_pivot_thresh,
            panel_size=panel_size,
            relax=relax,
        )

        setup_seconds = ctypes.c_double()
        factor_seconds = ctypes.c_double()
        info = ctypes.c_int()

        handle = lib.slumt_factor(
            nprocs,
            A.shape[0],
            A.nnz,
            A.data,
            A.indices,
            A.indptr,
            ctypes.byref(config),
            ctypes.byref(setup_seconds),
            ctypes.byref(factor_seconds),
            ctypes.byref(info),
        )

        if not handle:
            raise RuntimeError(
                f"SuperLU_MT factorization failed with info={info.value}"
            )

        if info.value != 0:
            lib.slumt_free(handle)
            raise RuntimeError(
                f"SuperLU_MT pdgstrf failed with info={info.value}"
            )

        return cls(
            lib=lib,
            handle=handle,
            n=A.shape[0],
            setup_seconds=setup_seconds.value,
            factor_seconds=factor_seconds.value,
        )

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self._handle is None:
            raise RuntimeError("SuperLU_MT factor has been closed")

        b = np.asarray(b, dtype=np.float64)

        # For now deliberately support one RHS only. Multiple RHS requires
        # respecting SuperLU's column-major dense-matrix layout.
        if b.ndim != 1 or b.shape[0] != self.n:
            raise ValueError(
                f"expected RHS with shape ({self.n},), got {b.shape}"
            )

        b = np.ascontiguousarray(b)
        x = np.empty(self.n, dtype=np.float64)

        solve_seconds = ctypes.c_double()

        info = self._lib.slumt_solve(
            self._handle,
            b,
            x,
            1,
            ctypes.byref(solve_seconds),
        )

        if info != 0:
            raise RuntimeError(
                f"SuperLU_MT dgstrs failed with info={info}"
            )

        self.solve_seconds = solve_seconds.value
        return x

    def close(self) -> None:
        if self._handle is not None:
            self._lib.slumt_free(self._handle)
            self._handle = None

    def __enter__(self) -> "SuperLUMTFactor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()