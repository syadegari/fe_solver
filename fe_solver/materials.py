from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .types import (
    EvaluationStatus,
    FailureKind,
    MaterialDefinition,
    MaterialModel,
    MaterialRequest,
    MaterialResponse,
    ModelError,
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


def evaluate_material_point(definition: MaterialDefinition, request: MaterialRequest) -> MaterialResponse:
    return definition.model.update(request)
