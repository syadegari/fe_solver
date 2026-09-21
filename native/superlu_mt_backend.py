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

_FLOAT64_ARRAY_P = np.ctypeslib.ndpointer(
    dtype=np.float64,
    ndim=1,
    flags="C_CONTIGUOUS",
)

_INT32_ARRAY_P = np.ctypeslib.ndpointer(
    dtype=np.int32,
    ndim=1,
    flags="C_CONTIGUOUS",
)


def load_superlu_mt(path: str | Path) -> ctypes.CDLL:
    """Load the native SuperLU_MT bridge and declare its ctypes ABI."""
    lib = ctypes.CDLL(str(Path(path)))

    lib.slumt_factor.argtypes = [
        _C_INT,                              # nprocs
        _C_INT,                              # n
        _C_INT,                              # nnz
        _FLOAT64_ARRAY_P,                    # data
        _INT32_ARRAY_P,                      # indices
        _INT32_ARRAY_P,                      # indptr
        ctypes.POINTER(SuperLUMTConfig),     # config
        _C_DOUBLE_P,                         # setup_seconds
        _C_DOUBLE_P,                         # factor_seconds
        _C_INT_P,                            # info_out
    ]
    lib.slumt_factor.restype = ctypes.c_void_p

    lib.slumt_solve.argtypes = [
        ctypes.c_void_p,
        _FLOAT64_ARRAY_P,
        _FLOAT64_ARRAY_P,
        _C_INT,
        _C_DOUBLE_P,
    ]
    lib.slumt_solve.restype = ctypes.c_int

    lib.slumt_free.argtypes = [ctypes.c_void_p]
    lib.slumt_free.restype = None

    return lib


def _validate_csc(A: sparse.spmatrix) -> None:
    """Validate the native bridge input contract without converting the matrix."""
    if not sparse.issparse(A) or A.format != "csc":
        raise TypeError("SuperLU_MT backend requires a CSC sparse matrix")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"SuperLU_MT requires a square matrix, got shape {A.shape}")

    # In the FE solver, K and C are formed through COO -> CSR conversion and
    # the KKT matrix is then built with sparse.bmat(..., format="csc"). Those
    # construction paths produce canonical compressed matrices (sorted indices,
    # no duplicate entries), so no canonicalization belongs in this bridge.
    # Keep the check because this wrapper can also be called independently and
    # raw CSR/CSC constructors can produce non-canonical matrices.
    if not A.has_canonical_format:
        raise ValueError(
            "SuperLU_MT backend requires canonical CSC "
            "(sorted indices and no duplicate entries)"
        )

    int32_max = np.iinfo(np.int32).max
    if A.shape[0] > int32_max or A.nnz > int32_max:
        raise OverflowError("matrix is too large for this 32-bit SuperLU_MT build")

    if A.data.dtype != np.float64:
        raise TypeError(f"matrix data must have dtype float64, got {A.data.dtype}")
    if A.indices.dtype != np.int32:
        raise TypeError(f"matrix indices must have dtype int32, got {A.indices.dtype}")
    if A.indptr.dtype != np.int32:
        raise TypeError(f"matrix indptr must have dtype int32, got {A.indptr.dtype}")

    if not A.data.flags.c_contiguous:
        raise ValueError("matrix data must be C-contiguous")
    if not A.indices.flags.c_contiguous:
        raise ValueError("matrix indices must be C-contiguous")
    if not A.indptr.flags.c_contiguous:
        raise ValueError("matrix indptr must be C-contiguous")


class SuperLUMTFactor:
    def __init__(
        self,
        lib: ctypes.CDLL,
        handle: int,
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
        lib: ctypes.CDLL,
        A: sparse.spmatrix,
        *,
        nprocs: int,
        ordering: str = "COLAMD",
        diag_pivot_thresh: float = 1.0,
        panel_size: int = -1,
        relax: int = -1,
    ) -> "SuperLUMTFactor":
        """Factor a canonical float64/int32 CSC matrix without converting it."""
        _validate_csc(A)

        # Ordering is expressed as a Python string, so mapping it to the C enum
        # is a Python-side responsibility. Native numeric option validation is
        # intentionally left to slumt_factor(), which is the authority for its
        # own API contract.
        try:
            permc_spec = ORDERINGS[ordering]
        except KeyError as exc:
            raise ValueError(f"unsupported SuperLU_MT ordering {ordering!r}") from exc

        config = SuperLUMTConfig(
            permc_spec=permc_spec,
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
        """Solve for one float64, C-contiguous RHS vector without converting it."""
        if self._handle is None:
            raise RuntimeError("SuperLU_MT factor has been closed")

        if not isinstance(b, np.ndarray):
            raise TypeError("RHS must be a NumPy array")
        if b.ndim != 1 or b.shape[0] != self.n:
            raise ValueError(f"expected RHS with shape ({self.n},), got {b.shape}")
        if b.dtype != np.float64:
            raise TypeError(f"RHS must have dtype float64, got {b.dtype}")
        if not b.flags.c_contiguous:
            raise ValueError("RHS must be C-contiguous")

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
            raise RuntimeError(f"SuperLU_MT dgstrs failed with info={info}")

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

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
