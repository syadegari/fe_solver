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

    for pindex, entry in enumerate(constraints.get("periodic_rve", [])):
        macro_data = entry["macro_F"]
        if len(macro_data) != 3 or any(len(row) != 3 for row in macro_data):
            raise ModelError("periodic macro_F must be a 3-by-3 value-expression matrix")
        macro = [[value_expression(macro_data[i][j]) for j in range(3)] for i in range(3)]
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
                def periodic_rhs(t: float, row=c, dx=delta_X.copy(), expressions=macro) -> float:
                    Fbar = np.array(
                        [[expressions[i][j].evaluate(t, deck.curves) for j in range(3)] for i in range(3)]
                    )
                    return float(((Fbar - np.eye(3)) @ dx)[row])
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

