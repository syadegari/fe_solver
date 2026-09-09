"""Array-only finite-strain J2 kernel with optional Numba compilation."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


J2_OK = 0
J2_INVALID_KINEMATICS = 1
J2_INVALID_STATE = 2
J2_NONPOSITIVE_STATE = 3
J2_NONUNIMODULAR_STATE = 4
J2_LOCAL_NONCONVERGENCE = 5
J2_NONPOSITIVE_RADIAL_SCALE = 6

_SQRT_TWO_THIRDS = float(np.sqrt(2.0 / 3.0))


def j2_update_kernel(
    F: np.ndarray,
    state_n: np.ndarray,
    shear_modulus: float,
    bulk_modulus: float,
    initial_yield_stress: float,
    linear_hardening_modulus: float,
    saturation_increment: float,
    saturation_rate: float,
    need_tangent: bool,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return status, ``P``, flattened ``dP/dF``, and trial state.

    This function deliberately accepts and returns only numeric scalars and
    arrays.  The public material routine owns named state/property access and
    converts the integer status into the common material API.  The same
    function is executed by CPython for the reference path and compiled in
    Numba nopython mode for the feasibility path.
    """
    P = np.zeros((3, 3))
    A_flat = np.empty(81) if need_tangent else np.empty(0)
    state_trial = state_n.copy()

    J = float(np.linalg.det(F))
    if not np.isfinite(J) or J <= 0.0:
        return J2_INVALID_KINEMATICS, P, A_flat, state_trial
    Finv = np.linalg.inv(F)

    Cp_inv = np.empty((3, 3))
    Cp_inv[0, 0] = state_n[0]
    Cp_inv[1, 1] = state_n[1]
    Cp_inv[2, 2] = state_n[2]
    Cp_inv[0, 1] = Cp_inv[1, 0] = state_n[3]
    Cp_inv[1, 2] = Cp_inv[2, 1] = state_n[4]
    Cp_inv[0, 2] = Cp_inv[2, 0] = state_n[5]
    ep_n = float(state_n[6])
    Cp_inv = 0.5 * (Cp_inv + Cp_inv.T)
    if not np.all(np.isfinite(Cp_inv)) or not np.isfinite(ep_n) or ep_n < 0.0:
        return J2_INVALID_STATE, P, A_flat, state_trial
    eigenvalues = np.linalg.eigvalsh(Cp_inv)
    det_cp_inv = float(np.linalg.det(Cp_inv))
    if eigenvalues[0] <= 0.0 or det_cp_inv <= 0.0:
        return J2_NONPOSITIVE_STATE, P, A_flat, state_trial
    if abs(det_cp_inv - 1.0) > 1.0e-8:
        return J2_NONUNIMODULAR_STATE, P, A_flat, state_trial

    identity = np.eye(3)
    Jm23 = J ** (-2.0 / 3.0)
    b_trial = Jm23 * F @ Cp_inv @ F.T
    b_trial = 0.5 * (b_trial + b_trial.T)
    dev_b_trial = b_trial - np.trace(b_trial) * identity / 3.0
    s_trial = shear_modulus * dev_b_trial
    norm_trial = float(np.linalg.norm(s_trial))

    exponential_n = float(np.exp(-saturation_rate * ep_n))
    yield_n = (
        initial_yield_stress
        + linear_hardening_modulus * ep_n
        + saturation_increment * (1.0 - exponential_n)
    )
    trial_function = norm_trial - _SQRT_TWO_THIRDS * yield_n
    scale = max(norm_trial, yield_n, shear_modulus, 1.0)
    plastic = trial_function > 1.0e-12 * scale
    mu_bar = shear_modulus * float(np.trace(b_trial)) / 3.0
    delta_gamma = 0.0
    beta = 1.0
    ep_np1 = ep_n

    if plastic:
        delta_gamma = max(0.0, trial_function / (2.0 * mu_bar))
        converged = False
        for _ in range(30):
            ep = ep_n + _SQRT_TWO_THIRDS * delta_gamma
            exponential = float(np.exp(-saturation_rate * ep))
            yield_stress = (
                initial_yield_stress
                + linear_hardening_modulus * ep
                + saturation_increment * (1.0 - exponential)
            )
            hardening_slope = (
                linear_hardening_modulus
                + saturation_increment * saturation_rate * exponential
            )
            residual = (
                norm_trial
                - 2.0 * mu_bar * delta_gamma
                - _SQRT_TWO_THIRDS * yield_stress
            )
            derivative = -(2.0 * mu_bar + (2.0 / 3.0) * hardening_slope)
            if abs(residual) <= 1.0e-12 * scale:
                converged = True
                break
            delta_gamma = max(0.0, delta_gamma - residual / derivative)
        if not converged:
            return J2_LOCAL_NONCONVERGENCE, P, A_flat, state_trial
        ep_np1 = ep_n + _SQRT_TWO_THIRDS * delta_gamma
        beta = 1.0 - 2.0 * mu_bar * delta_gamma / norm_trial
        if not np.isfinite(beta) or beta <= 0.0:
            return J2_NONPOSITIVE_RADIAL_SCALE, P, A_flat, state_trial

    s = beta * s_trial
    tau = bulk_modulus * J * (J - 1.0) * identity + s
    P = tau @ Finv.T

    if plastic:
        dev_b = s / shear_modulus
        dev_eigenvalues = np.linalg.eigvalsh(0.5 * (dev_b + dev_b.T))
        lower = max(0.0, -float(dev_eigenvalues[0])) + 1.0e-14
        upper = max(1.0, lower * 2.0)
        while (
            (dev_eigenvalues[0] + upper)
            * (dev_eigenvalues[1] + upper)
            * (dev_eigenvalues[2] + upper)
            - 1.0
        ) < 0.0:
            upper *= 2.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            determinant_residual = (
                (dev_eigenvalues[0] + middle)
                * (dev_eigenvalues[1] + middle)
                * (dev_eigenvalues[2] + middle)
                - 1.0
            )
            if determinant_residual > 0.0:
                upper = middle
            else:
                lower = middle
        spherical = 0.5 * (lower + upper)
        b_np1 = dev_b + spherical * identity
        Cp_inv_np1 = J ** (2.0 / 3.0) * Finv @ b_np1 @ Finv.T
        Cp_inv_np1 = 0.5 * (Cp_inv_np1 + Cp_inv_np1.T)
        state_trial[0] = Cp_inv_np1[0, 0]
        state_trial[1] = Cp_inv_np1[1, 1]
        state_trial[2] = Cp_inv_np1[2, 2]
        state_trial[3] = Cp_inv_np1[0, 1]
        state_trial[4] = Cp_inv_np1[1, 2]
        state_trial[5] = Cp_inv_np1[0, 2]
        state_trial[6] = ep_np1

    if need_tangent:
        A = A_flat.reshape((3, 3, 3, 3))
        exponential = float(np.exp(-saturation_rate * ep_np1))
        hardening_slope = (
            linear_hardening_modulus
            + saturation_increment * saturation_rate * exponential
        )
        denominator = 2.0 * mu_bar + (2.0 / 3.0) * hardening_slope
        for j in range(3):
            for M in range(3):
                dF = np.zeros((3, 3))
                dF[j, M] = 1.0
                dJ = J * float(np.trace(Finv @ dF))
                dJm23 = -(2.0 / 3.0) * Jm23 * dJ / J
                db_trial = (
                    dJm23 * (F @ Cp_inv @ F.T)
                    + Jm23 * (dF @ Cp_inv @ F.T + F @ Cp_inv @ dF.T)
                )
                ds_trial = shear_modulus * (
                    db_trial - np.trace(db_trial) * identity / 3.0
                )
                if plastic:
                    dnorm = float(np.sum(s_trial * ds_trial)) / norm_trial
                    dmu_bar = shear_modulus * float(np.trace(db_trial)) / 3.0
                    dgamma = (
                        dnorm - 2.0 * delta_gamma * dmu_bar
                    ) / denominator
                    dbeta = -2.0 * (
                        (dmu_bar * delta_gamma + mu_bar * dgamma) / norm_trial
                        - mu_bar * delta_gamma * dnorm / norm_trial**2
                    )
                    ds = dbeta * s_trial + beta * ds_trial
                else:
                    ds = ds_trial
                dtau = bulk_modulus * (2.0 * J - 1.0) * dJ * identity + ds
                dFinvT = -Finv.T @ dF.T @ Finv.T
                A[:, :, j, M] = dtau @ Finv.T + tau @ dFinvT

    return J2_OK, P, A_flat, state_trial


try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised in installations without the extra
    _NUMBA_KERNEL: Callable[..., tuple[int, np.ndarray, np.ndarray, np.ndarray]] | None = None
else:
    _NUMBA_KERNEL = njit(cache=True, fastmath=False)(j2_update_kernel)


_ACTIVE_BACKEND = "python"


def available_j2_backends() -> tuple[str, ...]:
    return ("python", "numba") if _NUMBA_KERNEL is not None else ("python",)


def configure_j2_backend(name: str) -> None:
    if name not in ("python", "numba"):
        raise ValueError("J2 backend must be 'python' or 'numba'")
    if name == "numba" and _NUMBA_KERNEL is None:
        raise RuntimeError("the Numba J2 backend requires the optional numba package")
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = name


def active_j2_backend() -> str:
    return _ACTIVE_BACKEND


def evaluate_j2_kernel(
    F: np.ndarray,
    state_n: np.ndarray,
    shear_modulus: float,
    bulk_modulus: float,
    initial_yield_stress: float,
    linear_hardening_modulus: float,
    saturation_increment: float,
    saturation_rate: float,
    need_tangent: bool,
):
    kernel = j2_update_kernel if _ACTIVE_BACKEND == "python" else _NUMBA_KERNEL
    assert kernel is not None
    return kernel(
        F,
        state_n,
        shear_modulus,
        bulk_modulus,
        initial_yield_stress,
        linear_hardening_modulus,
        saturation_increment,
        saturation_rate,
        need_tangent,
    )


def numba_signatures() -> tuple[object, ...]:
    if _NUMBA_KERNEL is None:
        return ()
    return tuple(_NUMBA_KERNEL.nopython_signatures)
