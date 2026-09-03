from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .config import Deck, component_index, value_expression
from .mesh import Mesh
from .types import ModelError


@dataclass(frozen=True)
class ConstraintSystem:
    C: sparse.csr_matrix
    rhs_functions: tuple[Callable[[float], float], ...]
    names: tuple[str, ...]

    def rhs(self, t: float) -> np.ndarray:
        return np.asarray([fn(t) for fn in self.rhs_functions], dtype=float)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def macro_deformation_function(entry: dict, deck: Deck) -> Callable[[float], np.ndarray]:
    """Compile either an explicit matrix or a supported exact macroscopic path."""
    has_matrix = "macro_F" in entry
    has_path = "macro_deformation" in entry
    if has_matrix == has_path:
        raise ModelError("constraint requires exactly one of macro_F or macro_deformation")
    if has_matrix:
        macro_data = entry["macro_F"]
        if len(macro_data) != 3 or any(len(row) != 3 for row in macro_data):
            raise ModelError("macro_F must be a 3-by-3 value-expression matrix")
        expressions = [[value_expression(macro_data[i][j]) for j in range(3)] for i in range(3)]

        def explicit(t: float) -> np.ndarray:
            return np.asarray(
                [[expressions[i][j].evaluate(t, deck.curves) for j in range(3)] for i in range(3)],
                dtype=float,
            )

        return explicit

    path = entry["macro_deformation"]
    if not isinstance(path, dict) or "type" not in path:
        raise ModelError("macro_deformation requires a type")
    kind = str(path["type"])
    if kind == "isochoric_uniaxial":
        unknown = set(path) - {"type", "axis", "stretch"}
        if unknown:
            raise ModelError(f"unknown isochoric_uniaxial fields: {sorted(unknown)}")
        axis = component_index(str(path.get("axis", "x")))
        stretch = value_expression(path["stretch"])

        def isochoric_uniaxial(t: float) -> np.ndarray:
            axial = stretch.evaluate(t, deck.curves)
            if axial <= 0.0:
                raise ModelError("isochoric_uniaxial stretch must remain positive")
            Fbar = np.eye(3) * axial ** -0.5
            Fbar[axis, axis] = axial
            return Fbar

        return isochoric_uniaxial
    if kind == "simple_shear":
        unknown = set(path) - {"type", "direction", "normal", "amount"}
        if unknown:
            raise ModelError(f"unknown simple_shear fields: {sorted(unknown)}")
        direction = component_index(str(path["direction"]))
        normal = component_index(str(path["normal"]))
        if direction == normal:
            raise ModelError("simple_shear direction and normal must differ")
        amount = value_expression(path["amount"])

        def simple_shear(t: float) -> np.ndarray:
            Fbar = np.eye(3)
            Fbar[direction, normal] = amount.evaluate(t, deck.curves)
            return Fbar

        return simple_shear
    raise ModelError(f"unsupported macro_deformation type {kind!r}")


def build_constraints(deck: Deck, mesh: Mesh) -> ConstraintSystem:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[Callable[[float], float]] = []
    names: list[str] = []
    signatures: set[tuple[tuple[int, float], ...]] = set()

    def add_row(coefficients: dict[int, float], fn: Callable[[float], float], name: str) -> None:
        filtered = {int(k): float(v) for k, v in coefficients.items() if v != 0.0}
        if not filtered:
            raise ModelError(f"constraint {name!r} has an all-zero row")
        signature = tuple(sorted(filtered.items()))
        if signature in signatures:
            raise ModelError(f"constraint {name!r} duplicates an existing constraint row")
        signatures.add(signature)
        row = len(rhs)
        for col, value in filtered.items():
            rows.append(row)
            cols.append(col)
            vals.append(value)
        rhs.append(fn)
        names.append(name)

    constraints = deck.data.get("constraints", {})
    for entry in constraints.get("prescribed", []):
        region = str(entry["region"])
        if region not in mesh.node_groups:
            raise ModelError(f"prescribed constraint references unknown region {region!r}")
        expr = value_expression(entry["value"])
        for component in entry["components"]:
            c = component_index(str(component))
            for node in mesh.node_groups[region]:
                add_row(
                    {3 * int(node) + c: 1.0},
                    lambda t, e=expr: e.evaluate(t, deck.curves),
                    f"{entry.get('name', region)}:{int(node)}:{component}",
                )

    for entry in constraints.get("linear", []):
        coefficients: dict[int, float] = {}
        seen: set[tuple[int, int]] = set()
        for term in entry["terms"]:
            tag = int(term["node_tag"])
            if tag not in mesh.tag_to_index:
                raise ModelError(f"linear constraint references unknown node tag {tag}")
            c = component_index(str(term["component"]))
            key = (tag, c)
            if key in seen:
                raise ModelError(f"duplicate term in linear constraint {entry.get('name', '')!r}")
            seen.add(key)
            dof = 3 * mesh.tag_to_index[tag] + c
            coefficients[dof] = float(term["coefficient"])
        expr = value_expression(entry["rhs"])
        add_row(
            coefficients,
            lambda t, e=expr: e.evaluate(t, deck.curves),
            str(entry.get("name", "linear")),
        )

    for aindex, entry in enumerate(constraints.get("affine", [])):
        regions = [str(region) for region in entry["regions"]]
        if not regions:
            raise ModelError("affine constraint requires at least one region")
        unknown = [region for region in regions if region not in mesh.node_groups]
        if unknown:
            raise ModelError(f"affine constraint references unknown regions {unknown}")
        nodes = np.unique(np.concatenate([mesh.node_groups[region] for region in regions]))
        origin = np.asarray(entry.get("origin", [0.0, 0.0, 0.0]), dtype=float)
        if origin.shape != (3,):
            raise ModelError("affine constraint origin must contain three coordinates")
        macro = macro_deformation_function(entry, deck)
        for node_raw in nodes:
            node = int(node_raw)
            relative_position = mesh.X[node] - origin
            for component in range(3):
                def affine_rhs(
                    t: float,
                    row=component,
                    position=relative_position.copy(),
                    deformation=macro,
                ) -> float:
                    return float(((deformation(t) - np.eye(3)) @ position)[row])

                add_row(
                    {3 * node + component: 1.0},
                    affine_rhs,
                    f"affine:{aindex}:{node}:{component}",
                )

    for pindex, entry in enumerate(constraints.get("periodic_rve", [])):
        macro = macro_deformation_function(entry, deck)
        union = _UnionFind(len(mesh.X))
        retained: list[tuple[int, int]] = []
        for region in entry["slave_regions"]:
            if region not in mesh.periodic_maps:
                raise ModelError(f"periodic slave region {region!r} has no Gmsh periodic metadata")
            for mapping in mesh.periodic_maps[region]:
                for slave, master in zip(mapping.slave_nodes, mapping.master_nodes):
                    if union.union(int(slave), int(master)):
                        retained.append((int(slave), int(master)))
        for edge_index, (slave, master) in enumerate(retained):
            delta_X = mesh.X[slave] - mesh.X[master]
            for c in range(3):
                def periodic_rhs(t: float, row=c, dx=delta_X.copy(), deformation=macro) -> float:
                    return float(((deformation(t) - np.eye(3)) @ dx)[row])
                add_row(
                    {3 * slave + c: 1.0, 3 * master + c: -1.0},
                    periodic_rhs,
                    f"periodic:{pindex}:{edge_index}:{c}",
                )
        anchor_region = str(entry["anchor_region"])
        if anchor_region not in mesh.node_groups or len(mesh.node_groups[anchor_region]) != 1:
            raise ModelError(f"periodic anchor region {anchor_region!r} must resolve to exactly one node")
        anchor = int(mesh.node_groups[anchor_region][0])
        for c in range(3):
            add_row({3 * anchor + c: 1.0}, lambda t: 0.0, f"periodic-anchor:{pindex}:{c}")

    C = sparse.coo_matrix((vals, (rows, cols)), shape=(len(rhs), mesh.ndof)).tocsr()
    if C.shape[0]:
        gram = (C @ C.T).tocsc()
        try:
            splu(gram)
        except RuntimeError as exc:
            raise ModelError("constraint matrix does not have independent rows") from exc
    return ConstraintSystem(C, tuple(rhs), tuple(names))
