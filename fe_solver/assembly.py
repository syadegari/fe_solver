from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import Deck, component_index, value_expression
from .elements import evaluate_element
from .materials import material_definition
from .mesh import Mesh
from .quadrature import HEX20_POINTS, HEX20_WEIGHTS, HEX8_POINTS, HEX8_WEIGHTS
from .shape import hex20_shape, hex8_shape
from .types import (
    ElementRequest,
    FailureKind,
    GaussOutput,
    MaterialDefinition,
    MaterialInitRequest,
    ModelError,
    RecoverableError,
)


@dataclass
class ElementBlock:
    region: str
    formulation: str
    material: MaterialDefinition
    element_tags: np.ndarray
    connectivity: np.ndarray
    state_n: np.ndarray


@dataclass
class FEModel:
    mesh: Mesh
    blocks: list[ElementBlock]
    deck: Deck


@dataclass
class AssemblyResult:
    f_int: np.ndarray
    K: sparse.csr_matrix | None
    state_trial: list[np.ndarray]
    gauss_output: list[list[GaussOutput]]


def build_model(deck: Deck, mesh: Mesh) -> FEModel:
    materials: dict[str, MaterialDefinition] = {}
    for entry in deck.data["materials"]:
        definition = material_definition(entry)
        if definition.name in materials:
            raise ModelError(f"duplicate material name {definition.name!r}")
        materials[definition.name] = definition
    claimed: dict[int, str] = {}
    blocks: list[ElementBlock] = []
    t0 = float(deck.data["analysis"]["t_start"])
    for assignment in deck.data["element_assignments"]:
        region = str(assignment["region"])
        formulation = str(assignment["formulation"])
        if formulation not in ("hex8", "hex8_fbar", "hex20"):
            raise ModelError(f"unsupported element formulation {formulation!r}")
        if region not in mesh.volume_groups:
            raise ModelError(f"element assignment references unknown volume region {region!r}")
        material_name = str(assignment["material"])
        if material_name not in materials:
            raise ModelError(f"element assignment references unknown material {material_name!r}")
        tags = mesh.volume_groups[region]
        connections: list[np.ndarray] = []
        for tag_raw in tags:
            tag = int(tag_raw)
            if tag in claimed:
                raise ModelError(f"element {tag} is ambiguously assigned by {claimed[tag]!r} and {region!r}")
            claimed[tag] = region
            element = mesh.elements[tag]
            expected = "hex8" if formulation == "hex8_fbar" else formulation
            if element.topology != expected:
                raise ModelError(
                    f"region {region!r} assigns {formulation} to a {element.topology} mesh element"
                )
            connections.append(element.connectivity)
        connectivity = np.asarray(connections, dtype=np.int64)
        if formulation == "hex20":
            ngauss = 27
            init_shape, init_points = hex20_shape, HEX20_POINTS
        else:
            ngauss = 8
            init_shape, init_points = hex8_shape, HEX8_POINTS
        nstate = materials[material_name].state_layout.n_state
        state = np.empty((len(tags), ngauss, nstate))
        material = materials[material_name]
        if nstate:
            assert material.model.initialize is not None
            for e, conn in enumerate(connectivity):
                for g, xi in enumerate(init_points):
                    N, _ = init_shape(xi)
                    init = material.model.initialize(
                        MaterialInitRequest(material.properties, None, N @ mesh.X[conn], t0)
                    )
                    if not init.status.ok:
                        raise ModelError(init.status.message)
                    if init.state0.shape != (nstate,):
                        raise ModelError(
                            f"initializer for {material.model_root!r} returned the wrong state size"
                        )
                    state[e, g] = init.state0
        blocks.append(ElementBlock(region, formulation, materials[material_name], tags.copy(), connectivity, state))
    missing = set(mesh.elements) - set(claimed)
    if missing:
        raise ModelError(f"{len(missing)} solid elements have no element assignment")
    return FEModel(mesh, blocks, deck)


def element_dofs(connectivity: np.ndarray) -> np.ndarray:
    return (3 * connectivity[:, None] + np.arange(3)).ravel()


def assemble_internal(
    model: FEModel,
    u_n: np.ndarray,
    u_trial: np.ndarray,
    t_n: float,
    t_np1: float,
    need_tangent: bool,
) -> AssemblyResult:
    ndof = model.mesh.ndof
    f_int = np.zeros(ndof)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    states: list[np.ndarray] = []
    all_outputs: list[list[GaussOutput]] = []
    for block in model.blocks:
        state_trial = np.empty_like(block.state_n)
        block_outputs: list[GaussOutput] = []
        for e, conn in enumerate(block.connectivity):
            dofs = element_dofs(conn)
            response = evaluate_element(
                ElementRequest(
                    model.mesh.X[conn], u_n[dofs], u_trial[dofs], block.state_n[e], block.material,
                    None, t_n, t_np1, need_tangent, block.formulation,
                )
            )
            if not response.status.ok:
                if response.status.kind is FailureKind.FATAL:
                    raise ModelError(f"element {int(block.element_tags[e])}: {response.status.message}")
                raise RecoverableError(f"element {int(block.element_tags[e])}: {response.status.message}")
            f_int[dofs] += response.f_int
            state_trial[e] = response.state_trial
            if response.gauss_output is not None:
                block_outputs.append(response.gauss_output)
            if need_tangent:
                assert response.K is not None
                rows.append(np.repeat(dofs, len(dofs)))
                cols.append(np.tile(dofs, len(dofs)))
                values.append(response.K.ravel())
        states.append(state_trial)
        all_outputs.append(block_outputs)
    K = None
    if need_tangent:
        K = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)
        ).tocsr()
        K.sum_duplicates()
    return AssemblyResult(f_int, K, states, all_outputs)


def assemble_external(model: FEModel, t: float) -> np.ndarray:
    f_ext = np.zeros(model.mesh.ndof)
    loads = model.deck.data.get("loads", {})
    for entry in loads.get("nodal", []):
        region = str(entry["region"])
        if region not in model.mesh.node_groups:
            raise ModelError(f"nodal load references unknown region {region!r}")
        c = component_index(str(entry["component"]))
        amplitude = value_expression(entry["value"]).evaluate(t, model.deck.curves)
        f_ext[3 * model.mesh.node_groups[region] + c] += amplitude
    for entry in loads.get("body", []):
        region = str(entry["region"])
        vector = np.asarray(
            [value_expression(x).evaluate(t, model.deck.curves) for x in entry["vector"]], dtype=float
        )
        matching = [block for block in model.blocks if block.region == region]
        if not matching:
            raise ModelError(f"body load references unassigned region {region!r}")
        for block in matching:
            if block.formulation == "hex20":
                shape, points, weights = hex20_shape, HEX20_POINTS, HEX20_WEIGHTS
            else:
                shape, points, weights = hex8_shape, HEX8_POINTS, HEX8_WEIGHTS
            for conn in block.connectivity:
                local = np.zeros((len(conn), 3))
                X = model.mesh.X[conn]
                for xi, weight in zip(points, weights):
                    N, dN = shape(xi)
                    detJ0 = float(np.linalg.det(X.T @ dN))
                    if detJ0 <= 0.0:
                        raise ModelError("nonpositive reference Jacobian during body-force assembly")
                    local += N[:, None] * vector * detJ0 * weight
                f_ext[element_dofs(conn)] += local.ravel()
    return f_ext


def commit_trial_states(model: FEModel, state_trial: list[np.ndarray]) -> None:
    if len(state_trial) != len(model.blocks):
        raise ValueError("material-state block count mismatch")
    for block, trial in zip(model.blocks, state_trial):
        if trial.shape != block.state_n.shape:
            raise ValueError("material-state shape mismatch")
    for block, trial in zip(model.blocks, state_trial):
        block.state_n[...] = trial
