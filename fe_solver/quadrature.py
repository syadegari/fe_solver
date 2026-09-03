from __future__ import annotations

import itertools

import numpy as np


def tensor_product_gauss(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order == 2:
        x = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w = np.ones(2)
    elif order == 3:
        x = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(f"unsupported Gauss order {order}")
    points: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for i, j, k in itertools.product(range(order), repeat=3):
        points.append((x[i], x[j], x[k]))
        weights.append(w[i] * w[j] * w[k])
    return np.asarray(points), np.asarray(weights)


HEX8_POINTS, HEX8_WEIGHTS = tensor_product_gauss(2)
HEX20_POINTS, HEX20_WEIGHTS = tensor_product_gauss(3)

