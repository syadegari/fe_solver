from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .j2_kernel import (
    J2_INVALID_KINEMATICS,
    J2_INVALID_STATE,
    J2_LOCAL_NONCONVERGENCE,
    J2_NONPOSITIVE_RADIAL_SCALE,
    J2_NONPOSITIVE_STATE,
    J2_NONUNIMODULAR_STATE,
    J2_OK,
    evaluate_j2_kernel,
)
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

_J2_FAILURE_MESSAGES = {
    J2_INVALID_KINEMATICS: "trial det(F) is nonpositive",
    J2_INVALID_STATE: "non-finite or negative committed J2 state",
    J2_NONPOSITIVE_STATE: "committed inverse plastic metric is not positive definite",
    J2_NONUNIMODULAR_STATE: "committed inverse plastic metric is not unimodular",
    J2_LOCAL_NONCONVERGENCE: "J2 plastic multiplier solve did not converge",
    J2_NONPOSITIVE_RADIAL_SCALE: "J2 return produced a nonpositive radial scale",
}


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
    if request.state_n.layout != _J2_LAYOUT:
        return _j2_failure(request, "invalid committed J2 state layout")
    try:
        properties = (
            float(request.properties["shear_modulus"]),
            float(request.properties["bulk_modulus"]),
            float(request.properties["initial_yield_stress"]),
            float(request.properties["linear_hardening_modulus"]),
            float(request.properties["saturation_increment"]),
            float(request.properties["saturation_rate"]),
        )
        status, P, A_flat, state_values = evaluate_j2_kernel(
            F, request.state_n.values, *properties, request.need_tangent
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _j2_failure(request, f"invalid J2 input: {exc}")
    except np.linalg.LinAlgError:
        return _j2_failure(request, "singular J2 trial tensor")

    if status != J2_OK:
        return _j2_failure(
            request,
            _J2_FAILURE_MESSAGES.get(status, f"unknown J2 kernel status {status}"),
        )
    A = A_flat.reshape((3, 3, 3, 3)) if request.need_tangent else None
    return MaterialResponse(P, A, _J2_LAYOUT.view(state_values), EvaluationStatus())


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
