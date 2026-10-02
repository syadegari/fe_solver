"""Result-field conventions and material-state output selection."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np

from .types import MaterialDefinition, ModelError, StateField


TENSOR_COMPONENTS = ("11", "22", "33", "12", "23", "13")
_ROWS = [0, 1, 2, 0, 1, 0]
_COLS = [0, 1, 2, 1, 2, 2]

_STATE_SELECTOR = re.compile(
    r"^(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[(?P<components>[^\[\]]+)\])?$"
)


@dataclass(frozen=True)
class StateOutputField:
    """One resolved, material-scoped result field derived from packed state."""

    selector: str
    source: StateField
    name: str
    component_indices: tuple[int, ...] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self.component_indices is None:
            return self.source.shape
        return () if len(self.component_indices) == 1 else (len(self.component_indices),)

    @property
    def component_order(self) -> tuple[str, ...]:
        if self.component_indices is None:
            return self.source.component_order
        if len(self.component_indices) == 1:
            return ()
        return tuple(self.source.component_order[index] for index in self.component_indices)


def _resolve_component_indices(field: StateField, expression: str) -> tuple[int, ...]:
    if len(field.shape) != 1 or not field.component_order:
        raise ModelError(
            f"material state field {field.name!r} does not declare selectable components"
        )
    labels = field.component_order
    positions = {label: index for index, label in enumerate(labels)}
    if ".." in expression:
        if "," in expression or expression.count("..") != 1:
            raise ModelError(f"invalid material-state component range {expression!r}")
        first, last = (token.strip() for token in expression.split(".."))
        if first not in positions or last not in positions:
            raise ModelError(
                f"unknown component in {field.name!r}[{expression}]; "
                f"declared components are {list(labels)!r}"
            )
        start, stop = positions[first], positions[last]
        if stop < start:
            raise ModelError(
                f"material-state component range {field.name!r}[{expression}] is reversed"
            )
        return tuple(range(start, stop + 1))

    selected = tuple(token.strip() for token in expression.split(","))
    if any(not token for token in selected):
        raise ModelError(f"invalid material-state component selection {expression!r}")
    unknown = [token for token in selected if token not in positions]
    if unknown:
        raise ModelError(
            f"unknown component(s) {unknown!r} for material state field {field.name!r}; "
            f"declared components are {list(labels)!r}"
        )
    if len(selected) != len(set(selected)):
        raise ModelError(f"duplicate component in {field.name!r}[{expression}]")
    return tuple(positions[token] for token in selected)


def resolve_material_state_output(
    output: Mapping[str, object],
    materials: Mapping[str, MaterialDefinition],
) -> dict[str, tuple[StateOutputField, ...]]:
    """Resolve the opt-in deck selectors against registered state layouts."""
    raw_entries = output.get("material_state", [])
    if not isinstance(raw_entries, list):
        raise ModelError("output.material_state must be an array of tables")
    resolved: dict[str, list[StateOutputField]] = {name: [] for name in materials}
    names_by_material: dict[str, set[str]] = {name: set() for name in materials}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ModelError("each output.material_state entry must be a table")
        unknown_keys = set(raw) - {"material", "select", "name"}
        if unknown_keys:
            raise ModelError(
                f"unknown output.material_state fields: {sorted(unknown_keys)}"
            )
        missing_keys = {"material", "select"} - set(raw)
        if missing_keys:
            raise ModelError(
                f"output.material_state entry is missing fields: {sorted(missing_keys)}"
            )
        material_name = str(raw["material"])
        try:
            material = materials[material_name]
        except KeyError as exc:
            raise ModelError(
                f"material-state output references unknown material {material_name!r}"
            ) from exc
        selector = str(raw["select"]).strip()
        match = _STATE_SELECTOR.fullmatch(selector)
        if match is None:
            raise ModelError(f"invalid material-state selector {selector!r}")
        source_name = match.group("field")
        fields = {field.name: field for field in material.state_layout.fields}
        try:
            source = fields[source_name]
        except KeyError as exc:
            raise ModelError(
                f"material {material_name!r} has no state field {source_name!r}"
            ) from exc

        component_expression = match.group("components")
        component_indices = (
            None
            if component_expression is None
            else _resolve_component_indices(source, component_expression)
        )
        if component_indices is not None and len(component_indices) > 1 and "name" not in raw:
            raise ModelError(
                f"multi-component material-state selector {selector!r} requires a name"
            )
        default_name = selector if component_indices is not None else source.name
        name = str(raw.get("name", default_name)).strip()
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise ModelError(f"invalid material-state output name {name!r}")
        if name in names_by_material[material_name]:
            raise ModelError(
                f"duplicate material-state output name {name!r} for material {material_name!r}"
            )
        names_by_material[material_name].add(name)
        resolved[material_name].append(
            StateOutputField(selector, source, name, component_indices)
        )
    return {name: tuple(fields) for name, fields in resolved.items()}


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
