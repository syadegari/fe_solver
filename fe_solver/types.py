from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping

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
class StateField:
    name: str
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or any(int(n) <= 0 for n in self.shape):
            raise ValueError("invalid material state field")

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=int)) if self.shape else 1


@dataclass(frozen=True)
class StateLayout:
    fields: tuple[StateField, ...] = ()

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("material state field names must be unique")

    @property
    def n_state(self) -> int:
        return sum(field.size for field in self.fields)

    def field_slice(self, name: str) -> tuple[slice, tuple[int, ...]]:
        offset = 0
        for field in self.fields:
            next_offset = offset + field.size
            if field.name == name:
                return slice(offset, next_offset), field.shape
            offset = next_offset
        raise KeyError(name)

    def view(self, values: np.ndarray) -> "MaterialStateView":
        return MaterialStateView(self, values)


@dataclass(frozen=True)
class MaterialStateView(Mapping[str, np.ndarray | float]):
    layout: StateLayout
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (self.layout.n_state,):
            raise ValueError("material state vector does not match its declared layout")
        object.__setattr__(self, "values", values)

    def __getitem__(self, name: str) -> np.ndarray | float:
        field_slice, shape = self.layout.field_slice(name)
        value = self.values[field_slice]
        return value.reshape(shape) if shape else float(value[0])

    def __iter__(self) -> Iterator[str]:
        return (field.name for field in self.layout.fields)

    def __len__(self) -> int:
        return len(self.layout.fields)


PropertyValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
MaterialRoutine = Callable[[Any], Any]


@dataclass(frozen=True)
class MaterialModel:
    root: str
    update: MaterialRoutine
    initialize: MaterialRoutine | None
    validate_properties: PropertyValidator
    state_layout: StateLayout = StateLayout()

    def __post_init__(self) -> None:
        if not self.root.isidentifier():
            raise ValueError(f"invalid material model root {self.root!r}")
        if self.update.__name__ != f"update_{self.root}":
            raise ValueError(f"material update must be named update_{self.root}")
        if self.initialize is not None and self.initialize.__name__ != f"init_{self.root}":
            raise ValueError(f"material initializer must be named init_{self.root}")
        if self.state_layout.n_state and self.initialize is None:
            raise ValueError("a history-dependent material model requires an initializer")


@dataclass(frozen=True)
class MaterialDefinition:
    name: str
    model: MaterialModel
    properties: Mapping[str, Any]

    @property
    def state_layout(self) -> StateLayout:
        return self.model.state_layout

    @property
    def model_root(self) -> str:
        return self.model.root


@dataclass(frozen=True)
class MaterialInitRequest:
    properties: Mapping[str, Any]
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
    state_n: MaterialStateView
    properties: Mapping[str, Any]
    point_properties: object | None
    t_n: float
    t_np1: float
    need_tangent: bool


@dataclass
class MaterialResponse:
    P: np.ndarray
    A_alg: np.ndarray | None
    state_trial: MaterialStateView
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
