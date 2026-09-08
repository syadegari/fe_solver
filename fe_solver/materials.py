from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .types import (
    EvaluationStatus,
    FailureKind,
    MaterialDefinition,
    MaterialInitRequest,
    MaterialInitResponse,
    MaterialModel,
    MaterialRequest,
    MaterialResponse,
    ModelError,
    StateField,
    StateLayout,
)


_SYMMETRIC_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
_SQRT_TWO_THIRDS = float(np.sqrt(2.0 / 3.0))


def _pack_symmetric(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[i, j] for i, j in _SYMMETRIC_PAIRS], dtype=float)


def _unpack_symmetric(values: np.ndarray) -> np.ndarray:
    tensor = np.empty((3, 3), dtype=float)
    for value, (i, j) in zip(np.asarray(values, dtype=float), _SYMMETRIC_PAIRS):
        tensor[i, j] = tensor[j, i] = value
    return tensor


def _validate_neo_hook(properties: Mapping[str, object]) -> Mapping[str, float]:
    unknown = set(properties) - {"mu", "kappa"}
    if unknown:
        raise ModelError(f"unknown neo_hook properties: {sorted(unknown)}")
    mu = float(properties.get("mu", 0.0))
    kappa = float(properties.get("kappa", 0.0))
    if mu <= 0.0 or kappa <= 0.0:
        raise ModelError("neo_hook requires mu > 0 and kappa > 0")
    return MappingProxyType({"mu": mu, "kappa": kappa})


def update_neo_hook(request: MaterialRequest) -> MaterialResponse:
    F = np.asarray(request.F_np1, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n,
            EvaluationStatus(FailureKind.RECOVERABLE, "non-finite trial deformation gradient"),
        )
    J = float(np.linalg.det(F))
    if not np.isfinite(J) or J <= 0.0:
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n,
            EvaluationStatus(FailureKind.RECOVERABLE, f"trial det(F) is nonpositive: {J}"),
        )
    try:
        FinvT = np.linalg.inv(F).T
    except np.linalg.LinAlgError:
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n,
            EvaluationStatus(FailureKind.RECOVERABLE, "singular trial deformation gradient"),
        )
    mu = float(request.properties["mu"])
    kappa = float(request.properties["kappa"])
    logJ = float(np.log(J))
    P = mu * (F - FinvT) + kappa * logJ * FinvT
    A = None
    if request.need_tangent:
        delta = np.eye(3)
        A = (
            mu * np.einsum("ij,IJ->iIjJ", delta, delta)
            + kappa * np.einsum("iI,jJ->iIjJ", FinvT, FinvT)
            + (mu - kappa * logJ) * np.einsum("iJ,jI->iIjJ", FinvT, FinvT)
        )
    return MaterialResponse(P, A, request.state_n, EvaluationStatus())


_J2_LAYOUT = StateLayout(
    (
        StateField("plastic_metric_inverse", (6,), ("11", "22", "33", "12", "23", "13")),
        StateField("equivalent_plastic_strain"),
    )
)


def _validate_j2_plasticity(properties: Mapping[str, object]) -> Mapping[str, float]:
    names = {
        "shear_modulus",
        "bulk_modulus",
        "initial_yield_stress",
        "linear_hardening_modulus",
        "saturation_increment",
        "saturation_rate",
    }
    unknown = set(properties) - names
    if unknown:
        raise ModelError(f"unknown j2_plasticity properties: {sorted(unknown)}")
    missing = names - set(properties)
    if missing:
        raise ModelError(f"missing j2_plasticity properties: {sorted(missing)}")
    try:
        values = {name: float(properties[name]) for name in names}
    except (TypeError, ValueError) as exc:
        raise ModelError("j2_plasticity properties must be numeric") from exc
    if not all(np.isfinite(value) for value in values.values()):
        raise ModelError("j2_plasticity properties must be finite")
    if values["shear_modulus"] <= 0.0 or values["bulk_modulus"] <= 0.0:
        raise ModelError("j2_plasticity requires positive shear_modulus and bulk_modulus")
    if values["initial_yield_stress"] <= 0.0:
        raise ModelError("j2_plasticity requires positive initial_yield_stress")
    if any(
        values[name] < 0.0
        for name in ("linear_hardening_modulus", "saturation_increment", "saturation_rate")
    ):
        raise ModelError("j2_plasticity hardening parameters must be nonnegative")
    return MappingProxyType(values)


def init_j2_plasticity(_request: MaterialInitRequest) -> MaterialInitResponse:
    state = np.zeros(_J2_LAYOUT.n_state)
    state[:3] = 1.0
    return MaterialInitResponse(state, EvaluationStatus())


def _j2_hardening(properties: Mapping[str, object], ep: float) -> tuple[float, float]:
    initial = float(properties["initial_yield_stress"])
    linear = float(properties["linear_hardening_modulus"])
    saturation = float(properties["saturation_increment"])
    rate = float(properties["saturation_rate"])
    exponential = float(np.exp(-rate * ep))
    value = initial + linear * ep + saturation * (1.0 - exponential)
    slope = linear + saturation * rate * exponential
    return value, slope


def _unit_determinant_spherical_part(deviator: np.ndarray) -> float:
    """Return c such that det(deviator + c I) = 1 and the tensor is positive definite."""
    eigenvalues = np.linalg.eigvalsh(0.5 * (deviator + deviator.T))
    lower = max(0.0, -float(eigenvalues[0])) + 1.0e-14

    def residual(value: float) -> float:
        return float(np.prod(eigenvalues + value) - 1.0)

    upper = max(1.0, lower * 2.0)
    while residual(upper) < 0.0:
        upper *= 2.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def _j2_failure(request: MaterialRequest, message: str) -> MaterialResponse:
    return MaterialResponse(
        np.zeros((3, 3)), None, request.state_n,
        EvaluationStatus(FailureKind.RECOVERABLE, message),
    )


def update_j2_plasticity(request: MaterialRequest) -> MaterialResponse:
    """Finite-strain, isochoric J2 return map from one committed endpoint.

    The isochoric elastic left metric is the predictor variable.  Its
    deviatoric Kirchhoff stress undergoes a radial return; the spherical
    part of the updated metric is reconstructed so its determinant is one.
    The nine tangent columns below are exact directional linearizations of
    this discrete return map, including the implicit plastic multiplier.
    """
    F = np.asarray(request.F_np1, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        return _j2_failure(request, "non-finite trial deformation gradient")
    J = float(np.linalg.det(F))
    if not np.isfinite(J) or J <= 0.0:
        return _j2_failure(request, f"trial det(F) is nonpositive: {J}")
    try:
        Finv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return _j2_failure(request, "singular trial deformation gradient")

    try:
        Cp_inv = _unpack_symmetric(request.state_n["plastic_metric_inverse"])
        ep_n = float(request.state_n["equivalent_plastic_strain"])
    except (KeyError, TypeError, ValueError) as exc:
        return _j2_failure(request, f"invalid committed J2 state: {exc}")
    Cp_inv = 0.5 * (Cp_inv + Cp_inv.T)
    if not np.all(np.isfinite(Cp_inv)) or not np.isfinite(ep_n) or ep_n < 0.0:
        return _j2_failure(request, "non-finite or negative committed J2 state")
    eig = np.linalg.eigvalsh(Cp_inv)
    det_cp_inv = float(np.linalg.det(Cp_inv))
    if eig[0] <= 0.0 or det_cp_inv <= 0.0:
        return _j2_failure(request, "committed inverse plastic metric is not positive definite")
    if abs(det_cp_inv - 1.0) > 1.0e-8:
        return _j2_failure(request, "committed inverse plastic metric is not unimodular")

    mu = float(request.properties["shear_modulus"])
    bulk = float(request.properties["bulk_modulus"])
    identity = np.eye(3)
    Jm23 = J ** (-2.0 / 3.0)
    b_trial = Jm23 * F @ Cp_inv @ F.T
    b_trial = 0.5 * (b_trial + b_trial.T)
    dev_b_trial = b_trial - np.trace(b_trial) * identity / 3.0
    s_trial = mu * dev_b_trial
    norm_trial = float(np.linalg.norm(s_trial))
    yield_n, _ = _j2_hardening(request.properties, ep_n)
    trial_function = norm_trial - _SQRT_TWO_THIRDS * yield_n
    scale = max(norm_trial, yield_n, mu, 1.0)
    plastic = trial_function > 1.0e-12 * scale
    mu_bar = mu * float(np.trace(b_trial)) / 3.0
    delta_gamma = 0.0
    beta = 1.0
    ep_np1 = ep_n

    if plastic:
        delta_gamma = max(0.0, trial_function / (2.0 * mu_bar))
        converged = False
        for _ in range(30):
            ep = ep_n + _SQRT_TWO_THIRDS * delta_gamma
            yield_stress, hardening_slope = _j2_hardening(request.properties, ep)
            residual = (
                norm_trial - 2.0 * mu_bar * delta_gamma
                - _SQRT_TWO_THIRDS * yield_stress
            )
            derivative = -(2.0 * mu_bar + (2.0 / 3.0) * hardening_slope)
            if abs(residual) <= 1.0e-12 * scale:
                converged = True
                break
            candidate = delta_gamma - residual / derivative
            delta_gamma = max(0.0, candidate)
        if not converged:
            return _j2_failure(request, "J2 plastic multiplier solve did not converge")
        ep_np1 = ep_n + _SQRT_TWO_THIRDS * delta_gamma
        beta = 1.0 - 2.0 * mu_bar * delta_gamma / norm_trial
        if not np.isfinite(beta) or beta <= 0.0:
            return _j2_failure(request, "J2 return produced a nonpositive radial scale")

    s = beta * s_trial
    tau = bulk * J * (J - 1.0) * identity + s
    P = tau @ Finv.T

    state_values = request.state_n.values.copy()
    if plastic:
        dev_b = s / mu
        spherical = _unit_determinant_spherical_part(dev_b)
        b_np1 = dev_b + spherical * identity
        Cp_inv_np1 = J ** (2.0 / 3.0) * Finv @ b_np1 @ Finv.T
        Cp_inv_np1 = 0.5 * (Cp_inv_np1 + Cp_inv_np1.T)
        state_values[:6] = _pack_symmetric(Cp_inv_np1)
        state_values[6] = ep_np1
    state_trial = _J2_LAYOUT.view(state_values)

    A = None
    if request.need_tangent:
        A = np.empty((3, 3, 3, 3))
        hardening_slope = _j2_hardening(request.properties, ep_np1)[1]
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
                ds_trial = mu * (db_trial - np.trace(db_trial) * identity / 3.0)
                if plastic:
                    dnorm = float(np.tensordot(s_trial, ds_trial, axes=2)) / norm_trial
                    dmu_bar = mu * float(np.trace(db_trial)) / 3.0
                    dgamma = (dnorm - 2.0 * delta_gamma * dmu_bar) / denominator
                    dbeta = -2.0 * (
                        (dmu_bar * delta_gamma + mu_bar * dgamma) / norm_trial
                        - mu_bar * delta_gamma * dnorm / norm_trial**2
                    )
                    ds = dbeta * s_trial + beta * ds_trial
                else:
                    ds = ds_trial
                dtau = bulk * (2.0 * J - 1.0) * dJ * identity + ds
                dFinvT = -Finv.T @ dF.T @ Finv.T
                A[:, :, j, M] = dtau @ Finv.T + tau @ dFinvT

    return MaterialResponse(P, A, state_trial, EvaluationStatus())


_MODELS: dict[str, MaterialModel] = {}


def register_material_model(model: MaterialModel) -> None:
    if model.root in _MODELS:
        raise ModelError(f"material model root {model.root!r} is already registered")
    _MODELS[model.root] = model


def get_material_model(root: str) -> MaterialModel:
    try:
        return _MODELS[root]
    except KeyError as exc:
        raise ModelError(f"unsupported material model {root!r}") from exc


register_material_model(
    MaterialModel(
        root="neo_hook",
        update=update_neo_hook,
        initialize=None,
        validate_properties=_validate_neo_hook,
        state_layout=StateLayout(),
    )
)

register_material_model(
    MaterialModel(
        root="j2_plasticity",
        update=update_j2_plasticity,
        initialize=init_j2_plasticity,
        validate_properties=_validate_j2_plasticity,
        state_layout=_J2_LAYOUT,
    )
)


def material_definition(data: dict) -> MaterialDefinition:
    root = str(data["model"])
    model = get_material_model(root)
    if "properties" in data and "parameters" in data:
        raise ModelError("a material cannot define both properties and legacy parameters")
    raw_properties = data.get("properties", data.get("parameters", {}))
    properties = MappingProxyType(dict(model.validate_properties(dict(raw_properties))))
    return MaterialDefinition(str(data["name"]), model, properties)


def neo_hook_definition(name: str, properties: Mapping[str, object]) -> MaterialDefinition:
    model = get_material_model("neo_hook")
    return MaterialDefinition(name, model, model.validate_properties(properties))


def j2_plasticity_definition(name: str, properties: Mapping[str, object]) -> MaterialDefinition:
    model = get_material_model("j2_plasticity")
    return MaterialDefinition(name, model, model.validate_properties(properties))


def evaluate_material_point(definition: MaterialDefinition, request: MaterialRequest) -> MaterialResponse:
    return definition.model.update(request)
