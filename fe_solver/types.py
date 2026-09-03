from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class FailureKind(str, Enum):
    OK = "ok"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


@dataclass(frozen=True)
class EvaluationStatus:
    kind: FailureKind = FailureKind.OK
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.kind is FailureKind.OK


@dataclass(frozen=True)
class StateLayout:
    n_state: int
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_state < 0 or len(self.names) not in (0, self.n_state):
            raise ValueError("invalid material state layout")


@dataclass(frozen=True)
class MaterialDefinition:
    name: str
    model: str
    parameters: Mapping[str, Any]
    state_layout: StateLayout


@dataclass(frozen=True)
class MaterialInitRequest:
    parameters: Mapping[str, Any]
    point_properties: object | None
    X: np.ndarray
    t0: float


@dataclass
class MaterialInitResponse:
    state0: np.ndarray
    status: EvaluationStatus


@dataclass(frozen=True)
class MaterialRequest:
    F_n: np.ndarray
    F_np1: np.ndarray
    state_n: np.ndarray
    parameters: Mapping[str, Any]
    point_properties: object | None
    t_n: float
    t_np1: float
    need_tangent: bool


@dataclass
class MaterialResponse:
    P: np.ndarray
    A_alg: np.ndarray | None
    state_trial: np.ndarray
    status: EvaluationStatus


@dataclass(frozen=True)
class ElementRequest:
    X_e: np.ndarray
    u_e_n: np.ndarray
    u_e_trial: np.ndarray
    state_e_n: np.ndarray
    material: MaterialDefinition
    point_properties: object | None
    t_n: float
    t_np1: float
    need_tangent: bool
    formulation: str


@dataclass
class GaussOutput:
    F_raw: list[np.ndarray] = field(default_factory=list)
    J_raw: list[float] = field(default_factory=list)
    F_material: list[np.ndarray] = field(default_factory=list)
    J_material: list[float] = field(default_factory=list)
    P_material: list[np.ndarray] = field(default_factory=list)
    P_effective: list[np.ndarray] = field(default_factory=list)
    cauchy_stress: list[np.ndarray] = field(default_factory=list)


@dataclass
class ElementResponse:
    f_int: np.ndarray
    K: np.ndarray | None
    state_trial: np.ndarray
    status: EvaluationStatus
    gauss_output: GaussOutput | None = None


class ModelError(RuntimeError):
    """Fatal model/setup error."""


class RecoverableError(RuntimeError):
    """Failure for which the increment controller may request a cutback."""

