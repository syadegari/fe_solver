"""Standalone drivers for prescribed deformation-gradient material paths."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .materials import evaluate_material_point
from .tangents import truesdell_voigt
from .types import (
    FailureKind,
    MaterialDefinition,
    MaterialInitRequest,
    MaterialRequest,
    ModelError,
    RecoverableError,
)


DeformationPath = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class MaterialPointHistory:
    path_parameter: np.ndarray
    time: np.ndarray
    F: np.ndarray
    P: np.ndarray
    kirchhoff_stress: np.ndarray
    cauchy_stress: np.ndarray
    green_lagrange_strain: np.ndarray
    euler_almansi_strain: np.ndarray
    material_log_strain: np.ndarray
    spatial_log_strain: np.ndarray
    state: np.ndarray
    A_alg: np.ndarray | None
    spatial_truesdell_tangent: np.ndarray | None


def linear_endpoint_path(F_target: np.ndarray) -> DeformationPath:
    """Return the component-wise path from identity to ``F_target``."""
    target = np.asarray(F_target, dtype=float)
    if target.shape != (3, 3) or not np.all(np.isfinite(target)):
        raise ValueError("target deformation gradient must be a finite 3x3 matrix")

    def path(parameter: float) -> np.ndarray:
        return np.eye(3) + float(parameter) * (target - np.eye(3))

    return path


def _symmetric_log(tensor: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (tensor + tensor.T))
    if values[0] <= 0.0:
        raise ModelError("strain logarithm received a non-positive-definite tensor")
    return (vectors * np.log(values)) @ vectors.T


def run_material_path(
    material: MaterialDefinition,
    deformation: DeformationPath | np.ndarray,
    number_of_steps: int,
    *,
    t_end: float = 1.0,
    X: np.ndarray | None = None,
    point_properties: object | None = None,
    need_tangent: bool = True,
) -> MaterialPointHistory:
    """Initialize once, then commit updates along a prescribed ``F(s)`` path.

    A 3x3 ``deformation`` value selects component-wise interpolation from the
    identity. A callable permits physically meaningful nonlinear paths.
    """
    if number_of_steps < 1:
        raise ValueError("number_of_steps must be positive")
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive")
    path = linear_endpoint_path(deformation) if isinstance(deformation, np.ndarray) else deformation
    if not callable(path):
        raise TypeError("deformation must be a 3x3 target or a callable path")
    location = np.zeros(3) if X is None else np.asarray(X, dtype=float)
    if location.shape != (3,):
        raise ValueError("X must have shape (3,)")

    layout = material.state_layout
    if material.model.initialize is None:
        state_values = np.empty(0)
    else:
        initialized = material.model.initialize(
            MaterialInitRequest(material.properties, point_properties, location, 0.0)
        )
        if not initialized.status.ok:
            raise ModelError(initialized.status.message)
        state_values = np.asarray(initialized.state0, dtype=float).copy()
    if state_values.shape != (layout.n_state,):
        raise ModelError("material initializer returned the wrong state size")

    parameters = np.linspace(0.0, 1.0, number_of_steps + 1)
    times = t_end * parameters
    F_values = np.empty((number_of_steps + 1, 3, 3))
    P_values = np.empty_like(F_values)
    tau_values = np.empty_like(F_values)
    sigma_values = np.empty_like(F_values)
    green_values = np.empty_like(F_values)
    almansi_values = np.empty_like(F_values)
    material_log_values = np.empty_like(F_values)
    spatial_log_values = np.empty_like(F_values)
    states = np.empty((number_of_steps + 1, layout.n_state))
    A_values = np.empty((number_of_steps + 1, 3, 3, 3, 3)) if need_tangent else None
    D_values = np.empty((number_of_steps + 1, 6, 6)) if need_tangent else None

    F_n = np.eye(3)
    for index, (parameter, time) in enumerate(zip(parameters, times)):
        F = np.asarray(path(float(parameter)), dtype=float)
        if F.shape != (3, 3) or not np.all(np.isfinite(F)):
            raise ModelError(f"deformation path returned invalid F at step {index}")
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ModelError(f"deformation path returned nonpositive det(F) at step {index}")
        state_n = layout.view(state_values)
        response = evaluate_material_point(
            material,
            MaterialRequest(
                F_n, F, state_n, material.properties, point_properties,
                float(times[index - 1]) if index else 0.0, float(time), need_tangent,
            ),
        )
        if not response.status.ok:
            error = RecoverableError if response.status.kind is FailureKind.RECOVERABLE else ModelError
            raise error(f"material path failed at step {index}: {response.status.message}")
        if response.state_trial.values.shape != (layout.n_state,):
            raise ModelError("material update returned the wrong state size")

        tau = response.P @ F.T
        sigma = tau / J
        C = F.T @ F
        b = F @ F.T
        Finv = np.linalg.inv(F)
        F_values[index] = F
        P_values[index] = response.P
        tau_values[index] = tau
        sigma_values[index] = sigma
        green_values[index] = 0.5 * (C - np.eye(3))
        almansi_values[index] = 0.5 * (np.eye(3) - Finv.T @ Finv)
        material_log_values[index] = 0.5 * _symmetric_log(C)
        spatial_log_values[index] = 0.5 * _symmetric_log(b)
        states[index] = response.state_trial.values
        if need_tangent:
            assert response.A_alg is not None and A_values is not None and D_values is not None
            A_values[index] = response.A_alg
            D_values[index] = truesdell_voigt(response.A_alg, F, J, sigma)

        state_values = response.state_trial.values.copy()
        F_n = F.copy()

    return MaterialPointHistory(
        parameters, times, F_values, P_values, tau_values, sigma_values,
        green_values, almansi_values, material_log_values, spatial_log_values,
        states, A_values, D_values,
    )
