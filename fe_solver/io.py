from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import AssemblyResult, FEModel
from .mesh import Mesh
from .types import ModelError


def model_identity(model: FEModel) -> str:
    digest = hashlib.sha256()
    digest.update(model.mesh.X.tobytes())
    digest.update(model.mesh.node_tags.tobytes())
    for block in model.blocks:
        digest.update(block.region.encode())
        digest.update(block.formulation.encode())
        digest.update(block.material.name.encode())
        digest.update(block.material.model.encode())
        digest.update(block.element_tags.tobytes())
        digest.update(block.connectivity.tobytes())
        digest.update(str(block.state_n.shape).encode())
    identity_tables = {
        key: model.deck.data.get(key)
        for key in ("curves", "materials", "element_assignments", "constraints", "loads")
    }
    digest.update(json.dumps(identity_tables, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _vtu_connectivity(mesh: Mesh) -> tuple[list[np.ndarray], list[int]]:
    cells: list[np.ndarray] = []
    types: list[int] = []
    # Gmsh and VTK share corner ordering but not the Hex20 edge ordering.
    hex20_to_vtk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15])
    for tag in sorted(mesh.elements):
        element = mesh.elements[tag]
        if element.topology == "hex8":
            cells.append(element.connectivity)
            types.append(12)
        else:
            cells.append(element.connectivity[hex20_to_vtk])
            types.append(25)
    return cells, types


def write_vtu(path: Path, mesh: Mesh, u: np.ndarray) -> None:
    cells, types = _vtu_connectivity(mesh)
    flat = np.concatenate(cells)
    offsets = np.cumsum([len(c) for c in cells])
    points = mesh.X
    displacement = u.reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        stream.write(f'<UnstructuredGrid><Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">\n')
        stream.write('<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(stream, points, fmt="%.16e")
        stream.write('</DataArray></Points>\n<Cells>\n')
        stream.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(stream, flat[None], fmt="%d")
        stream.write('</DataArray><DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(stream, offsets[None], fmt="%d")
        stream.write('</DataArray><DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(stream, np.asarray(types)[None], fmt="%d")
        stream.write('</DataArray></Cells>\n<PointData Vectors="displacement">\n')
        stream.write('<DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(stream, displacement, fmt="%.16e")
        stream.write('</DataArray></PointData>\n</Piece></UnstructuredGrid></VTKFile>\n')


def gauss_arrays(assembly: AssemblyResult, requested: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for b, outputs in enumerate(assembly.gauss_output):
        for name in requested:
            if not outputs:
                continue
            if not hasattr(outputs[0], name):
                raise ModelError(f"unknown requested Gauss field {name!r}")
            result[f"block_{b}_{name}"] = np.asarray([getattr(item, name) for item in outputs])
    return result


def write_accepted_output(
    directory: Path,
    index: int,
    t: float,
    u: np.ndarray,
    lambdas: np.ndarray,
    reaction: np.ndarray,
    assembly: AssemblyResult,
    model: FEModel,
    newton_history: list[dict],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    requested = list(model.deck.data["output"].get("write_gauss_fields", []))
    data = {
        "pseudo_time": np.asarray(t), "u": u, "lambda": lambdas,
        "constraint_reaction": reaction,
        **gauss_arrays(assembly, requested),
    }
    for b, block in enumerate(model.blocks):
        data[f"block_{b}_material_state"] = block.state_n
    np.savez_compressed(directory / f"state_{index:06d}.npz", **data)
    if model.deck.data["output"].get("write_vtu", False):
        write_vtu(directory / f"state_{index:06d}.vtu", model.mesh, u)
    if model.deck.data["output"].get("write_newton_history", False):
        with (directory / "newton_history.json").open("w", encoding="utf-8") as stream:
            json.dump(newton_history, stream, indent=2)


def write_restart(
    path: Path,
    model: FEModel,
    t_n: float,
    proposed_dt: float,
    u_n: np.ndarray,
    lambda_n: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1),
        "model_identity": np.asarray(model_identity(model)),
        "t_n": np.asarray(t_n),
        "proposed_dt": np.asarray(proposed_dt),
        "u_n": u_n,
        "lambda_n": lambda_n,
    }
    for b, block in enumerate(model.blocks):
        data[f"state_n_{b}"] = block.state_n
    np.savez_compressed(path, **data)


def load_restart(path: Path, model: FEModel) -> tuple[float, float, np.ndarray, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except OSError as exc:
        raise ModelError(f"cannot read restart file {path}") from exc
    with archive:
        if int(archive["schema_version"]) != 1:
            raise ModelError("unsupported restart schema version")
        if str(archive["model_identity"]) != model_identity(model):
            raise ModelError("restart mesh/assignment/material identity mismatch")
        u = np.asarray(archive["u_n"], dtype=float).copy()
        lambdas = np.asarray(archive["lambda_n"], dtype=float).copy()
        if u.shape != (model.mesh.ndof,):
            raise ModelError("restart displacement shape mismatch")
        for b, block in enumerate(model.blocks):
            state = np.asarray(archive[f"state_n_{b}"], dtype=float)
            if state.shape != block.state_n.shape:
                raise ModelError("restart material state layout mismatch")
            block.state_n[...] = state
        return float(archive["t_n"]), float(archive["proposed_dt"]), u, lambdas
