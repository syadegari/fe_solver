from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import h5py
import numpy as np

from .assembly import AssemblyResult, FEModel
from .output_fields import TENSOR_COMPONENTS, pack_symmetric
from .shape import hex20_shape, hex8_shape
from .types import ModelError


RESULT_SCHEMA_VERSION = 3
RESTART_SCHEMA_VERSION = 2


def model_identity(model: FEModel) -> str:
    digest = hashlib.sha256()
    digest.update(model.mesh.X.tobytes())
    digest.update(model.mesh.node_tags.tobytes())
    for block in model.blocks:
        digest.update(block.region.encode())
        digest.update(block.formulation.encode())
        digest.update(block.material.name.encode())
        digest.update(block.material.model_root.encode())
        digest.update(block.element_tags.tobytes())
        digest.update(block.connectivity.tobytes())
        digest.update(str(block.state_n.shape).encode())
    identity_tables = {
        key: model.deck.data.get(key)
        for key in ("curves", "materials", "element_assignments", "constraints", "loads")
    }
    digest.update(json.dumps(identity_tables, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda x: np.asarray(x).tolist())


def _center_sample(values: np.ndarray, formulation: str) -> np.ndarray:
    """Recover parent-centroid values without creating a constitutive point."""
    if formulation == "hex20":
        return values[:, 13].copy()
    return np.mean(values, axis=1)


@dataclass(frozen=True)
class CellFields:
    cauchy_stress: np.ndarray
    green_lagrange_strain: np.ndarray
    euler_almansi_strain: np.ndarray
    state: dict[str, np.ndarray]


def recover_cell_fields(model: FEModel, assembly: AssemblyResult, u: np.ndarray) -> list[CellFields]:
    fields: list[CellFields] = []
    x = model.mesh.X + u.reshape(-1, 3)
    for block_index, block in enumerate(model.blocks):
        stresses = np.asarray(
            [element.cauchy_stress for element in assembly.gauss_output[block_index]], dtype=float
        )
        stress_center = _center_sample(stresses, block.formulation)
        green = np.empty((len(block.connectivity), 3, 3))
        almansi = np.empty_like(green)
        shape = hex20_shape if block.formulation == "hex20" else hex8_shape
        _, dN = shape(np.zeros(3))
        for element_index, connectivity in enumerate(block.connectivity):
            X_e = model.mesh.X[connectivity]
            x_e = x[connectivity]
            F = (x_e.T @ dN) @ np.linalg.inv(X_e.T @ dN)
            green[element_index] = 0.5 * (F.T @ F - np.eye(3))
            Finv = np.linalg.inv(F)
            almansi[element_index] = 0.5 * (np.eye(3) - Finv.T @ Finv)
        packed_center = _center_sample(block.state_n, block.formulation)
        state: dict[str, np.ndarray] = {}
        for state_field in block.material.state_layout.fields:
            field_slice, field_shape = block.material.state_layout.field_slice(state_field.name)
            values = packed_center[:, field_slice]
            state[state_field.name] = values.reshape((len(block.connectivity), *field_shape))
        fields.append(CellFields(stress_center, green, almansi, state))
    return fields


def _create_time_dataset(group: h5py.Group, name: str, item_shape: tuple[int, ...]) -> h5py.Dataset:
    dataset = group.create_dataset(
        name,
        shape=(0, *item_shape),
        maxshape=(None, *item_shape),
        chunks=(1, *item_shape),
        dtype=np.float64,
        compression="gzip",
        shuffle=True,
    )
    dataset.attrs["time_dependent"] = True
    return dataset


class HDF5ResultWriter:
    """Append-only accepted-state database; incomplete tail rows are ignored."""

    def __init__(self, path: Path, model: FEModel, *, resume: bool = False):
        self.path = path
        self.model = model
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or not resume
        self.file = h5py.File(path, "w" if new_file else "a")
        if new_file:
            self._initialize()
        else:
            self._validate_and_truncate()

    def _initialize(self) -> None:
        root = self.file
        root.attrs["schema_version"] = RESULT_SCHEMA_VERSION
        root.attrs["model_identity"] = model_identity(self.model)
        root.attrs["git_commit"] = _git_commit(self.model.deck.path.parent)
        meta = root.create_group("meta")
        meta.attrs["analysis_name"] = str(self.model.deck.data["analysis"]["name"])
        meta.create_dataset("resolved_input_json", data=_json(self.model.deck.data), dtype=h5py.string_dtype())

        mesh = root.create_group("mesh")
        mesh.create_dataset("reference_coordinates", data=self.model.mesh.X)
        mesh.create_dataset("node_tags", data=self.model.mesh.node_tags)
        mesh_blocks = mesh.create_group("blocks")
        for index, block in enumerate(self.model.blocks):
            group = mesh_blocks.create_group(f"{index:04d}")
            group.attrs["region"] = block.region
            group.attrs["formulation"] = block.formulation
            group.attrs["material"] = block.material.name
            group.attrs["connectivity_ordering"] = "gmsh"
            group.create_dataset("element_tags", data=block.element_tags)
            group.create_dataset("connectivity", data=block.connectivity)

        materials = root.create_group("materials")
        seen: set[str] = set()
        for block in self.model.blocks:
            material = block.material
            if material.name in seen:
                continue
            group = materials.create_group(f"{len(seen):04d}")
            seen.add(material.name)
            group.attrs["name"] = material.name
            group.attrs["model"] = material.model_root
            group.attrs["properties_json"] = _json(dict(material.properties))
            group.attrs["state_layout_json"] = _json(
                [{"name": field.name, "shape": field.shape} for field in material.state_layout.fields]
            )

        curves = root.create_group("curves")
        for index, curve in enumerate(self.model.deck.curves.values()):
            group = curves.create_group(f"{index:04d}")
            group.attrs["name"] = curve.name
            group.create_dataset("time", data=curve.times)
            group.create_dataset("value", data=curve.values)

        results = root.create_group("results")
        results.attrs["n_complete_steps"] = 0
        _create_time_dataset(results, "time", ())
        nodal = results.create_group("nodal")
        _create_time_dataset(nodal, "displacement", (len(self.model.mesh.X), 3))
        reaction = _create_time_dataset(nodal, "constraint_reaction", (len(self.model.mesh.X), 3))
        reaction.attrs["sign_convention"] = "structure_on_constraint=-C.T@lambda"
        result_blocks = results.create_group("blocks")
        for index, block in enumerate(self.model.blocks):
            group = result_blocks.create_group(f"{index:04d}")
            group.attrs["region"] = block.region
            group.attrs["material"] = block.material.name
            shape = (len(block.connectivity), 6)
            stress = _create_time_dataset(group, "cauchy_stress", shape)
            stress.attrs["centering"] = "cell"
            stress.attrs["recovery"] = "central_gauss_point" if block.formulation == "hex20" else "gauss_interpolation"
            green = _create_time_dataset(group, "green_lagrange_strain", shape)
            green.attrs["frame"] = "material"
            green.attrs["centering"] = "cell"
            almansi = _create_time_dataset(group, "euler_almansi_strain", shape)
            almansi.attrs["frame"] = "spatial"
            almansi.attrs["centering"] = "cell"
            for tensor in (stress, green, almansi):
                tensor.attrs["component_order"] = np.asarray(TENSOR_COMPONENTS, dtype=h5py.string_dtype())
                tensor.attrs["shear_convention"] = "tensorial"
                tensor.attrs["shear_scale"] = 1.0
            state_group = group.create_group("state")
            for field in block.material.state_layout.fields:
                dataset = _create_time_dataset(
                    state_group, field.name, (len(block.connectivity), *field.shape)
                )
                dataset.attrs["centering"] = "cell"
                dataset.attrs["recovery"] = stress.attrs["recovery"]
        root.flush()

    def _time_datasets(self) -> list[h5py.Dataset]:
        datasets: list[h5py.Dataset] = []

        def collect(_name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset) and item.attrs.get("time_dependent", False):
                datasets.append(item)

        self.file["results"].visititems(collect)
        return datasets

    def _validate_and_truncate(self) -> None:
        if int(self.file.attrs.get("schema_version", -1)) != RESULT_SCHEMA_VERSION:
            raise ModelError("results database schema version mismatch")
        if str(self.file.attrs.get("model_identity", "")) != model_identity(self.model):
            raise ModelError("results database model identity mismatch")
        complete = int(self.file["results"].attrs["n_complete_steps"])
        for dataset in self._time_datasets():
            if dataset.shape[0] < complete:
                raise ModelError("results database has an incomplete committed prefix")
            if dataset.shape[0] != complete:
                dataset.resize(complete, axis=0)

    @property
    def n_complete_steps(self) -> int:
        return int(self.file["results"].attrs["n_complete_steps"])

    @property
    def last_time(self) -> float | None:
        count = self.n_complete_steps
        return float(self.file["results/time"][count - 1]) if count else None

    @staticmethod
    def _append(dataset: h5py.Dataset, index: int, value: np.ndarray | float) -> None:
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def append(self, t: float, u: np.ndarray, reaction: np.ndarray, assembly: AssemblyResult) -> None:
        if self.last_time is not None and abs(self.last_time - t) <= 1.0e-13 * max(1.0, abs(t)):
            return
        if self.last_time is not None and t < self.last_time:
            raise ModelError("results database times must be strictly increasing")
        index = self.n_complete_steps
        cell_fields = recover_cell_fields(self.model, assembly, u)
        self._append(self.file["results/time"], index, t)
        self._append(self.file["results/nodal/displacement"], index, u.reshape(-1, 3))
        self._append(self.file["results/nodal/constraint_reaction"], index, reaction.reshape(-1, 3))
        for block_index, fields in enumerate(cell_fields):
            group = self.file[f"results/blocks/{block_index:04d}"]
            self._append(group["cauchy_stress"], index, pack_symmetric(fields.cauchy_stress))
            self._append(group["green_lagrange_strain"], index, pack_symmetric(fields.green_lagrange_strain))
            self._append(group["euler_almansi_strain"], index, pack_symmetric(fields.euler_almansi_strain))
            for name, values in fields.state.items():
                self._append(group[f"state/{name}"], index, values)
        self.file.flush()
        self.file["results"].attrs.modify("n_complete_steps", index + 1)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "HDF5ResultWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def write_run_log(path: Path, increments: list[Any], newton_history: list[dict], verification: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "increments": [asdict(item) for item in increments],
        "newton_history": newton_history,
        "verification": verification,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def write_restart(
    path: Path,
    model: FEModel,
    t_n: float,
    proposed_dt: float,
    u_n: np.ndarray,
    lambda_n: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as archive:
        archive.attrs["schema_version"] = RESTART_SCHEMA_VERSION
        archive.attrs["model_identity"] = model_identity(model)
        archive.attrs["t_n"] = t_n
        archive.attrs["proposed_dt"] = proposed_dt
        archive.create_dataset("u_n", data=u_n)
        archive.create_dataset("lambda_n", data=lambda_n)
        states = archive.create_group("material_state")
        for block_index, block in enumerate(model.blocks):
            states.create_dataset(f"{block_index:04d}", data=block.state_n)


def load_restart(path: Path, model: FEModel) -> tuple[float, float, np.ndarray, np.ndarray]:
    try:
        archive = h5py.File(path, "r")
    except OSError as exc:
        raise ModelError(f"cannot read restart file {path}") from exc
    with archive:
        if int(archive.attrs.get("schema_version", -1)) != RESTART_SCHEMA_VERSION:
            raise ModelError("unsupported restart schema version")
        if str(archive.attrs.get("model_identity", "")) != model_identity(model):
            raise ModelError("restart mesh/assignment/material identity mismatch")
        u = np.asarray(archive["u_n"], dtype=float)
        lambdas = np.asarray(archive["lambda_n"], dtype=float)
        if u.shape != (model.mesh.ndof,):
            raise ModelError("restart displacement shape mismatch")
        for block_index, block in enumerate(model.blocks):
            state = np.asarray(archive[f"material_state/{block_index:04d}"], dtype=float)
            if state.shape != block.state_n.shape:
                raise ModelError("restart material state layout mismatch")
            block.state_n[...] = state
        return (
            float(archive.attrs["t_n"]), float(archive.attrs["proposed_dt"]),
            u.copy(), lambdas.copy(),
        )
