"""Symmetric-tensor reporting convention (not engineering-shear assembly vectors)."""
from __future__ import annotations

import numpy as np


TENSOR_COMPONENTS = ("11", "22", "33", "12", "23", "13")
_ROWS = [0, 1, 2, 0, 1, 0]
_COLS = [0, 1, 2, 1, 2, 2]


def pack_symmetric(values: np.ndarray) -> np.ndarray:
    """Report [..., 3, 3] tensors as [..., 6], without scaling shear entries."""
    values = np.asarray(values)
    if values.shape[-2:] != (3, 3):
        raise ValueError("expected tensors with trailing shape (3, 3)")
    return values[..., _ROWS, _COLS]


def unpack_symmetric(values: np.ndarray) -> np.ndarray:
    """Reconstruct symmetric tensors for postprocessing contractions/rotations."""
    values = np.asarray(values)
    if values.shape[-1:] != (6,):
        raise ValueError("expected six tensor components")
    tensors = np.empty((*values.shape[:-1], 3, 3), dtype=values.dtype)
    tensors[..., _ROWS, _COLS] = values
    tensors[..., _COLS, _ROWS] = values
    return tensors
