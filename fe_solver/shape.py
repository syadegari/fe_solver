from __future__ import annotations

import numpy as np


HEX8_PARENT_NODES = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=float,
)

# This is the ordering returned by Gmsh element type 17.
HEX20_PARENT_NODES = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        [0, -1, -1], [-1, 0, -1], [-1, -1, 0], [1, 0, -1],
        [1, -1, 0], [0, 1, -1], [1, 1, 0], [-1, 1, 0],
        [0, -1, 1], [-1, 0, 1], [1, 0, 1], [0, 1, 1],
    ],
    dtype=float,
)


def hex8_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi = np.asarray(xi, dtype=float)
    factors = 1.0 + HEX8_PARENT_NODES * xi
    N = np.prod(factors, axis=1) / 8.0
    dN = np.empty((8, 3))
    for a, signs in enumerate(HEX8_PARENT_NODES):
        dN[a, 0] = signs[0] * factors[a, 1] * factors[a, 2] / 8.0
        dN[a, 1] = signs[1] * factors[a, 0] * factors[a, 2] / 8.0
        dN[a, 2] = signs[2] * factors[a, 0] * factors[a, 1] / 8.0
    return N, dN


def hex20_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, s, t = np.asarray(xi, dtype=float)
    N = np.empty(20)
    dN = np.empty((20, 3))
    for a, (ra, sa, ta) in enumerate(HEX20_PARENT_NODES):
        zero = np.flatnonzero(np.array([ra, sa, ta]) == 0.0)
        if len(zero) == 0:
            fr, fs, ft = 1 + r * ra, 1 + s * sa, 1 + t * ta
            h = r * ra + s * sa + t * ta - 2
            N[a] = fr * fs * ft * h / 8.0
            dN[a, 0] = ra * fs * ft * (h + fr) / 8.0
            dN[a, 1] = sa * fr * ft * (h + fs) / 8.0
            dN[a, 2] = ta * fr * fs * (h + ft) / 8.0
        elif zero[0] == 0:
            fs, ft = 1 + s * sa, 1 + t * ta
            N[a] = (1 - r * r) * fs * ft / 4.0
            dN[a] = (-2 * r * fs * ft, (1 - r * r) * sa * ft, (1 - r * r) * fs * ta)
            dN[a] /= 4.0
        elif zero[0] == 1:
            fr, ft = 1 + r * ra, 1 + t * ta
            N[a] = (1 - s * s) * fr * ft / 4.0
            dN[a] = ((1 - s * s) * ra * ft, -2 * s * fr * ft, (1 - s * s) * fr * ta)
            dN[a] /= 4.0
        else:
            fr, fs = 1 + r * ra, 1 + s * sa
            N[a] = (1 - t * t) * fr * fs / 4.0
            dN[a] = ((1 - t * t) * ra * fs, (1 - t * t) * fr * sa, -2 * t * fr * fs)
            dN[a] /= 4.0
    return N, dN


def shape_for(formulation: str):
    if formulation in ("hex8", "hex8_fbar"):
        return hex8_shape
    if formulation == "hex20":
        return hex20_shape
    raise ValueError(f"unsupported formulation {formulation!r}")

