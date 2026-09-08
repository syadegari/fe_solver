"""Transform material endpoint tangents into FE spatial tangent matrices."""
from __future__ import annotations

import numpy as np


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def truesdell_voigt(
    A: np.ndarray, F: np.ndarray, J: float, sigma: np.ndarray,
) -> np.ndarray:
    """Return the 6x6 Truesdell tangent using engineering-shear columns."""
    a_pf = np.einsum("iIkK,jI,mK->ijkm", A, F, F) / J
    identity = np.eye(3)
    c = 0.5 * (
        a_pf + a_pf.transpose(0, 1, 3, 2)
        - np.einsum("ik,mj->ijkm", identity, sigma)
        - np.einsum("im,kj->ijkm", identity, sigma)
    )
    D = np.empty((6, 6))
    for row, (i, j) in enumerate(VOIGT_PAIRS):
        for col, (k, m) in enumerate(VOIGT_PAIRS):
            D[row, col] = (
                c[i, j, k, k]
                if k == m
                else 0.5 * (c[i, j, k, m] + c[i, j, m, k])
            )
    return D
