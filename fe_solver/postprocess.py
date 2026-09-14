from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np

from .io import RESULT_SCHEMA_VERSION
from .output_fields import TENSOR_COMPONENTS
from .types import ModelError


# XDMF/VTK edge order differs from Gmsh type 17 after the eight corner nodes.
_GMSH_TO_XDMF_HEX20 = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15],
    dtype=np.int64,
)


def _data_item(
    parent: ET.Element,
    dimensions: tuple[int, ...],
    reference: str,
    *,
    number_type: str = "Float",
    precision: str = "8",
) -> ET.Element:
    item = ET.SubElement(
        parent,
        "DataItem",
        Dimensions=" ".join(map(str, dimensions)),
        NumberType=number_type,
        Precision=precision,
        Format="HDF",
    )
    item.text = reference
    return item


def _view_path(dataset_path: str, step: int) -> str:
    return f"/steps/{step:06d}{dataset_path}"


def _create_time_views(
    visual: h5py.File, dataset: h5py.Dataset, complete: int, source_ref: str,
) -> None:
    """Expose time slices as ordinary HDF datasets without copying values.

    ParaView's XDMF3 reader does not reliably resolve XML HyperSlab items.
    HDF5 virtual datasets perform the same selection inside the HDF5 reader.
    """
    source = h5py.VirtualSource(source_ref, dataset.name, shape=dataset.shape)
    for step in range(complete):
        layout = h5py.VirtualLayout(shape=dataset.shape[1:], dtype=dataset.dtype)
        layout[...] = source[step, ...]
        visual.create_virtual_dataset(_view_path(dataset.name, step), layout)


def _time_slice(
    parent: ET.Element,
    full_shape: tuple[int, ...],
    step: int,
    reference: str,
) -> None:
    item_shape = full_shape[1:]
    file_ref, dataset_path = reference.split(":", 1)
    _data_item(parent, item_shape, f"{file_ref}:{_view_path(dataset_path, step)}")


def _attribute_type(shape: tuple[int, ...]) -> str:
    value_shape = shape[2:]
    if not value_shape or value_shape == (1,):
        return "Scalar"
    if value_shape == (3,):
        return "Vector"
    if value_shape == (3, 3):
        return "Tensor"
    return "Matrix"


def write_xdmf(database: str | Path, output: str | Path | None = None) -> Path:
    """Create one ParaView-readable temporal XDMF collection from a solver HDF5 database."""
    database = Path(database).resolve()
    output_path = Path(output).resolve() if output is not None else database.with_suffix(".xdmf")
    sidecar = output_path.with_suffix(".xdmf.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = h5py.File(database, "r")
    except OSError as exc:
        raise ModelError(f"cannot read results database {database}") from exc
    with source:
        detected_version = int(source.attrs.get("schema_version", -1))
        if detected_version != RESULT_SCHEMA_VERSION:
            raise ModelError(
                "unsupported results database schema version "
                f"{detected_version}; current postprocessing requires version "
                f"{RESULT_SCHEMA_VERSION}. Rerun the simulation with the current "
                "solver; results databases are not converted in place"
            )
        complete = int(source["results"].attrs.get("n_complete_steps", -1))
        times = np.asarray(source["results/time"][:complete], dtype=float)
        if complete < 1 or len(times) != complete:
            raise ModelError("results database contains no complete accepted state")
        coordinates = source["mesh/reference_coordinates"]
        node_count = int(coordinates.shape[0])
        block_names = sorted(source["mesh/blocks"].keys())
        source_ref = Path(os.path.relpath(database, output_path.parent)).as_posix()
        sidecar_ref = sidecar.name
        with h5py.File(sidecar, "w") as visual:
            for dataset in source["results/nodal"].values():
                _create_time_views(visual, dataset, complete, source_ref)
            for name in block_names:
                mesh_block = source[f"mesh/blocks/{name}"]
                connectivity = np.asarray(mesh_block["connectivity"], dtype=np.int64)
                formulation = str(mesh_block.attrs["formulation"])
                if formulation == "hex20":
                    connectivity = connectivity[:, _GMSH_TO_XDMF_HEX20]
                visual.create_dataset(f"connectivity/{name}", data=connectivity)
                result_block = source[f"results/blocks/{name}"]
                for field_name in ("cauchy_stress", "green_lagrange_strain", "euler_almansi_strain"):
                    _create_time_views(visual, result_block[field_name], complete, source_ref)
                if "state" in result_block:
                    for dataset in result_block["state"].values():
                        _create_time_views(visual, dataset, complete, source_ref)

        xdmf = ET.Element("Xdmf", Version="3.0")
        domain = ET.SubElement(xdmf, "Domain")
        spatial = ET.SubElement(
            domain,
            "Grid",
            Name="phase_blocks",
            GridType="Collection",
            CollectionType="Spatial",
        )
        for name in block_names:
            mesh_block = source[f"mesh/blocks/{name}"]
            result_block = source[f"results/blocks/{name}"]
            formulation = str(mesh_block.attrs["formulation"])
            cell_count, node_per_cell = map(int, mesh_block["connectivity"].shape)
            phase = ET.SubElement(
                spatial,
                "Grid",
                Name=f"{mesh_block.attrs['region']}:{mesh_block.attrs['material']}",
                GridType="Collection",
                CollectionType="Temporal",
            )
            for step, time in enumerate(times):
                grid = ET.SubElement(
                    phase,
                    "Grid",
                    Name="state",
                    GridType="Uniform",
                )
                ET.SubElement(grid, "Time", Value=f"{time:.17g}")
                topology_name = "Hexahedron_20" if formulation == "hex20" else "Hexahedron"
                topology = ET.SubElement(
                    grid, "Topology", TopologyType=topology_name, NumberOfElements=str(cell_count)
                )
                _data_item(
                    topology,
                    (cell_count, node_per_cell),
                    f"{sidecar_ref}:/connectivity/{name}",
                    number_type="Int",
                )
                geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
                _data_item(geometry, (node_count, 3), f"{source_ref}:/mesh/reference_coordinates")

                displacement = source["results/nodal/displacement"]
                attribute = ET.SubElement(
                    grid, "Attribute", Name="displacement", AttributeType="Vector", Center="Node"
                )
                _time_slice(attribute, tuple(displacement.shape), step, f"{sidecar_ref}:/results/nodal/displacement")

                reaction = source["results/nodal/constraint_reaction"]
                attribute = ET.SubElement(
                    grid, "Attribute", Name="constraint_reaction", AttributeType="Vector", Center="Node"
                )
                _time_slice(
                    attribute, tuple(reaction.shape), step,
                    f"{sidecar_ref}:/results/nodal/constraint_reaction",
                )
                for field_name in (
                    "cauchy_stress", "green_lagrange_strain", "euler_almansi_strain"
                ):
                    dataset = result_block[field_name]
                    attribute = ET.SubElement(
                        grid, "Attribute", Name=field_name, AttributeType="Matrix", Center="Cell"
                    )
                    # Generic six-component arrays preserve our ordering; do not
                    # apply a reader-specific Tensor6 convention or expand to nine.
                    ET.SubElement(attribute, "Information", Name="component_order", Value=",".join(TENSOR_COMPONENTS))
                    _time_slice(
                        attribute, tuple(dataset.shape), step,
                        f"{sidecar_ref}:/results/blocks/{name}/{field_name}",
                    )
                if "state" in result_block:
                    for state_name, dataset in result_block["state"].items():
                        attribute = ET.SubElement(
                            grid,
                            "Attribute",
                            Name=f"state_{state_name}",
                            AttributeType=_attribute_type(tuple(dataset.shape)),
                            Center="Cell",
                        )
                        if "component_order" in dataset.attrs:
                            labels = [
                                item.decode() if isinstance(item, bytes) else str(item)
                                for item in dataset.attrs["component_order"]
                            ]
                            ET.SubElement(
                                attribute,
                                "Information",
                                Name="component_order",
                                Value=",".join(labels),
                            )
                        _time_slice(
                            attribute, tuple(dataset.shape), step,
                            f"{sidecar_ref}:/results/blocks/{name}/state/{state_name}",
                        )
        ET.indent(xdmf, space="  ")
        ET.ElementTree(xdmf).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="write temporal XDMF from a finite-element run database")
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(write_xdmf(args.database, args.output))


if __name__ == "__main__":
    main()
