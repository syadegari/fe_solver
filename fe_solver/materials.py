from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .types import (
    EvaluationStatus,
    FailureKind,
    MaterialDefinition,
    MaterialInitRequest,
    MaterialInitResponse,
    MaterialRequest,
    MaterialResponse,
    ModelError,
    StateLayout,
)


def neo_hookean_definition(name: str, parameters: dict[str, float]) -> MaterialDefinition:
    mu = float(parameters.get("mu", 0.0))
    kappa = float(parameters.get("kappa", 0.0))
    if mu <= 0.0 or kappa <= 0.0:
        raise ModelError(f"neo-Hookean material {name!r} requires mu > 0 and kappa > 0")
    return MaterialDefinition(name, "neo_hookean", {"mu": mu, "kappa": kappa}, StateLayout(0))


def initialize_neo_hookean(request: MaterialInitRequest) -> MaterialInitResponse:
    return MaterialInitResponse(np.empty(0), EvaluationStatus())


def evaluate_neo_hookean(request: MaterialRequest) -> MaterialResponse:
    F = np.asarray(request.F_np1, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n.copy(),
            EvaluationStatus(FailureKind.RECOVERABLE, "non-finite trial deformation gradient"),
        )
    J = float(np.linalg.det(F))
    if not np.isfinite(J) or J <= 0.0:
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n.copy(),
            EvaluationStatus(FailureKind.RECOVERABLE, f"trial det(F) is nonpositive: {J}"),
        )
    try:
        FinvT = np.linalg.inv(F).T
    except np.linalg.LinAlgError:
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n.copy(),
            EvaluationStatus(FailureKind.RECOVERABLE, "singular trial deformation gradient"),
        )
    mu = float(request.parameters["mu"])
    kappa = float(request.parameters["kappa"])
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
    return MaterialResponse(P, A, request.state_n.copy(), EvaluationStatus())


_INITIALIZERS: dict[str, Callable[[MaterialInitRequest], MaterialInitResponse]] = {
    "neo_hookean": initialize_neo_hookean,
}
_EVALUATORS: dict[str, Callable[[MaterialRequest], MaterialResponse]] = {
    "neo_hookean": evaluate_neo_hookean,
}


def material_definition(data: dict) -> MaterialDefinition:
    model = str(data["model"])
    if model == "neo_hookean":
        return neo_hookean_definition(str(data["name"]), dict(data.get("parameters", {})))
    raise ModelError(f"unsupported material model {model!r}")


def initialize_material_state(definition: MaterialDefinition, request: MaterialInitRequest) -> MaterialInitResponse:
    return _INITIALIZERS[definition.model](request)


def evaluate_material_point(definition: MaterialDefinition, request: MaterialRequest) -> MaterialResponse:
    return _EVALUATORS[definition.model](request)
