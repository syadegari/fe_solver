"""Regenerate the frozen patch-test cases (housekeeping, not part of running).

Writes each case's mesh (.msh 4.1) and deck (.toml) into its subdirectory and
round-trips the result through the production reader, asserting it matches the
in-memory construction exactly.  The committed files are the inputs of record;
this script only exists so changes to them are traceable:

    conda run -n py3.14 python -m verification.patch_tests.generate_patch_cases
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fe_solver.mesh import Mesh
from fe_solver.shape import HEX20_PARENT_NODES, HEX8_PARENT_NODES
from verification.patch_tests.patch_cube import (
    CASES, FACES, SEED, build_patch_mesh, check_frozen_case, frozen_deck_text,
)


ROOT = Path(__file__).resolve().parent

# Entity tags: points 1..8, curves 1..12, surfaces 1..6 (xmin..zmax), volume 1.
# Physical tags follow the convention of the generated example meshes:
# solid = 1 (dim 3), xmin..zmax = 2..7 (dim 2).
_SURFACE_TAGS = {name: index + 1 for index, name in enumerate(FACES)}
_PHYSICAL_TAGS = {"solid": 1, **{name: index + 2 for index, name in enumerate(FACES)}}


def _point_tag(bits: tuple[int, int, int]) -> int:
    return 1 + bits[0] + 2 * bits[1] + 4 * bits[2]


def _curve_tag(axis: int, fixed_bits: tuple[int, int]) -> int:
    return 1 + 4 * axis + fixed_bits[0] + 2 * fixed_bits[1]


def _classify(reference_position: np.ndarray):
    """Return (dim, entity tag) for a node, from its reference coordinates."""
    bits = []
    fixed_axes = []
    for axis in range(3):
        value = reference_position[axis]
        if value == 0.0 or value == 1.0:
            fixed_axes.append(axis)
            bits.append(int(value))
    if len(fixed_axes) == 3:
        return 0, _point_tag(tuple(bits))
    if len(fixed_axes) == 2:
        free_axis = ({0, 1, 2} - set(fixed_axes)).pop()
        return 1, _curve_tag(free_axis, tuple(bits))
    if len(fixed_axes) == 1:
        axis = fixed_axes[0]
        return 2, _SURFACE_TAGS[FACES[2 * axis + bits[0]]]
    return 3, 1


def _boundary_quads(mesh: Mesh, reference: np.ndarray, topology: str):
    """Element faces lying on each cube face, as (surface tag, node-tag rows).

    Corner ordering is derived from the parent coordinates; for hex20 the four
    edge mid-nodes follow in the 8-node serendipity quad convention.
    """
    parents = HEX20_PARENT_NODES if topology == "hex20" else HEX8_PARENT_NODES
    quads: dict[int, list[list[int]]] = {tag: [] for tag in _SURFACE_TAGS.values()}
    for element in mesh.elements.values():
        conn = element.connectivity
        for axis in range(3):
            for sign, side in ((-1.0, 0), (1.0, 1)):
                local_corners = [
                    a for a in range(8) if parents[a][axis] == sign
                ]
                if any(
                    reference[conn[a]][axis] != float(side) for a in local_corners
                ):
                    continue
                other = [k for k in range(3) if k != axis]
                angles = np.arctan2(
                    [parents[a][other[1]] for a in local_corners],
                    [parents[a][other[0]] for a in local_corners],
                )
                ordered = [local_corners[i] for i in np.argsort(angles)]
                row = [int(conn[a]) + 1 for a in ordered]
                if topology == "hex20":
                    for first, second in zip(ordered, ordered[1:] + ordered[:1]):
                        target = 0.5 * (parents[first] + parents[second])
                        mid = next(
                            a for a in range(8, 20)
                            if np.array_equal(parents[a], target)
                        )
                        row.append(int(conn[mid]) + 1)
                quads[_SURFACE_TAGS[FACES[2 * axis + side]]].append(row)
    return quads


def write_patch_mesh(mesh: Mesh, reference: np.ndarray, topology: str, path: Path) -> None:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("patch_cube")

        for bits in [(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]:
            gmsh.model.addDiscreteEntity(0, _point_tag(bits))
        for axis in range(3):
            for b1 in (0, 1):
                for b0 in (0, 1):
                    fixed = (b0, b1)
                    start = [0, 0, 0]
                    end = [0, 0, 0]
                    end[axis] = 1
                    others = [k for k in range(3) if k != axis]
                    for position, bit in zip(others, fixed):
                        start[position] = bit
                        end[position] = bit
                    gmsh.model.addDiscreteEntity(
                        1, _curve_tag(axis, fixed),
                        [_point_tag(tuple(start)), -_point_tag(tuple(end))],
                    )
        for axis in range(3):
            for side in (0, 1):
                boundary = []
                others = [k for k in range(3) if k != axis]
                for curve_axis in others:
                    third = ({0, 1, 2} - {axis, curve_axis}).pop()
                    for bit in (0, 1):
                        fixed_axes = sorted([axis, third])
                        values = {axis: side, third: bit}
                        boundary.append(_curve_tag(
                            curve_axis, tuple(values[a] for a in fixed_axes)
                        ))
                gmsh.model.addDiscreteEntity(
                    2, _SURFACE_TAGS[FACES[2 * axis + side]], boundary
                )
        gmsh.model.addDiscreteEntity(3, 1, list(_SURFACE_TAGS.values()))

        blocks: dict[tuple[int, int], list[int]] = {}
        for node_index in range(len(mesh.X)):
            blocks.setdefault(_classify(reference[node_index]), []).append(node_index)
        for (dim, tag), node_indices in sorted(blocks.items()):
            gmsh.model.mesh.addNodes(
                dim, tag,
                [index + 1 for index in node_indices],
                np.asarray(mesh.X[node_indices], dtype=float).ravel(),
            )

        element_type = 17 if topology == "hex20" else 5
        tags = sorted(mesh.elements)
        gmsh.model.mesh.addElements(
            3, 1, [element_type], [tags],
            [[int(node) + 1 for tag in tags for node in mesh.elements[tag].connectivity]],
        )
        quad_type = 16 if topology == "hex20" else 3
        next_tag = 1001
        for surface_tag, rows in _boundary_quads(mesh, reference, topology).items():
            gmsh.model.mesh.addElements(
                2, surface_tag, [quad_type],
                [list(range(next_tag, next_tag + len(rows)))],
                [[node for row in rows for node in row]],
            )
            next_tag += len(rows)

        for name, tag in _PHYSICAL_TAGS.items():
            dim = 3 if name == "solid" else 2
            entity = 1 if name == "solid" else _SURFACE_TAGS[name]
            gmsh.model.addPhysicalGroup(dim, [entity], tag)
            gmsh.model.setPhysicalName(dim, tag, name)

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def main() -> None:
    written_meshes: set[str] = set()
    for formulation, (topology, mesh_file) in CASES.items():
        directory = ROOT / formulation
        directory.mkdir(exist_ok=True)
        mesh, reference = build_patch_mesh(topology, SEED)
        write_patch_mesh(mesh, reference, topology, directory / mesh_file)
        (directory / f"patch_{formulation}.toml").write_text(frozen_deck_text(formulation))
        check_frozen_case(directory, formulation)
        written_meshes.add(f"{formulation}/{mesh_file}")
        print(f"{formulation}: wrote {mesh_file} and patch_{formulation}.toml; round-trip OK")


if __name__ == "__main__":
    main()
