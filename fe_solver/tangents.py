"""Transform material endpoint tangents into FE spatial tangent matrices."""
from __future__ import annotations

import numpy as np
from numba import njit


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


@njit(cache=True, fastmath=False)
def truesdell_voigt(
    A: np.ndarray, F: np.ndarray, J: float, sigma: np.ndarray,
) -> np.ndarray:
    """Return the 6x6 Truesdell tangent using engineering-shear columns."""
    D = np.empty((6, 6))
    for row, (i, j) in enumerate(VOIGT_PAIRS):
        for col, (k, m) in enumerate(VOIGT_PAIRS):
            a_ijkm = 0.0
            a_ijmk = 0.0
            for I in range(3):
                for K in range(3):
                    a_ijkm += A[i, I, k, K] * F[j, I] * F[m, K]
                    a_ijmk += A[i, I, m, K] * F[j, I] * F[k, K]
            D[row, col] = 0.5 * (
                (a_ijkm + a_ijmk) / J
                - (sigma[m, j] if i == k else 0.0)
                - (sigma[k, j] if i == m else 0.0)
            )
    return D
