"""Generate a conforming periodic cube with a half-width central core."""
from __future__ import annotations

import argparse

import gmsh


def near(a: float, b: float, tol: float = 1.0e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def bounds(dim: int, tag: int) -> tuple[float, ...]:
    return tuple(float(value) for value in gmsh.model.getBoundingBox(dim, tag))


def translated_face_key(tag: int, axis: int) -> tuple[float, ...]:
    box = bounds(2, tag)
    values = [box[i] for i in range(6) if i not in (axis, axis + 3)]
    return tuple(round(value, 10) for value in values)


def classify_outer_faces() -> dict[str, list[int]]:
    groups = {name: [] for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    for _, tag in gmsh.model.getEntities(2):
        box = bounds(2, tag)
        for axis, low, high in ((0, "xmin", "xmax"), (1, "ymin", "ymax"), (2, "zmin", "zmax")):
            if near(box[axis], 0.0) and near(box[axis + 3], 0.0):
                groups[low].append(tag)
            elif near(box[axis], 1.0) and near(box[axis + 3], 1.0):
                groups[high].append(tag)
    if any(len(tags) != 9 for tags in groups.values()):
        raise RuntimeError(f"expected nine patches on each outer face, got {groups}")
    for axis, pair in enumerate((("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))):
        for name in pair:
            groups[name].sort(key=lambda tag, a=axis: translated_face_key(tag, a))
    return groups


def add_physical(dim: int, tags: list[int], name: str) -> None:
    physical = gmsh.model.addPhysicalGroup(dim, tags)
    gmsh.model.setPhysicalName(dim, physical, name)


def translation(axis: int) -> list[float]:
    transform = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    transform[4 * axis + 3] = 1.0
    return transform


def find_origin() -> int:
    points = []
    for _, tag in gmsh.model.getEntities(0):
        box = bounds(0, tag)
        if near(box[0], 0.0) and near(box[1], 0.0) and near(box[2], 0.0):
            points.append(tag)
    if len(points) != 1:
        raise RuntimeError(f"expected one origin point, got {points}")
    return points[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        help="output .msh path (defaults to a name derived from the resolution)",
    )
    parser.add_argument(
        "--elements-per-edge",
        type=int,
        default=8,
        help="total cube divisions per edge; must be a positive multiple of four",
    )
    args = parser.parse_args()
    elements_per_edge = int(args.elements_per_edge)
    if elements_per_edge <= 0 or elements_per_edge % 4:
        parser.error("--elements-per-edge must be a positive multiple of four")
    output = args.out or (
        f"heterogeneous_periodic_cube_{elements_per_edge}x"
        f"{elements_per_edge}x{elements_per_edge}.msh"
    )
    coordinates = (0.0, 0.25, 0.75, 1.0)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"heterogeneous_periodic_cube_{elements_per_edge}")
        boxes = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    boxes.append(
                        (3, gmsh.model.occ.addBox(
                            coordinates[i], coordinates[j], coordinates[k],
                            coordinates[i + 1] - coordinates[i],
                            coordinates[j + 1] - coordinates[j],
                            coordinates[k + 1] - coordinates[k],
                        ))
                    )
        gmsh.model.occ.fragment(boxes, [])
        gmsh.model.occ.synchronize()

        core: list[int] = []
        matrix: list[int] = []
        for _, tag in gmsh.model.getEntities(3):
            box = bounds(3, tag)
            if all(near(box[i], 0.25) and near(box[i + 3], 0.75) for i in range(3)):
                core.append(tag)
            else:
                matrix.append(tag)
        if len(core) != 1 or len(matrix) != 26:
            raise RuntimeError(f"unexpected volume partition: {len(core)=}, {len(matrix)=}")

        for _, tag in gmsh.model.getEntities(1):
            box = bounds(1, tag)
            length = max(box[i + 3] - box[i] for i in range(3))
            divisions = round(length * elements_per_edge)
            expected_divisions = {
                elements_per_edge // 4,
                elements_per_edge // 2,
            }
            if divisions not in expected_divisions or not near(
                length, divisions / elements_per_edge
            ):
                raise RuntimeError(f"unexpected curve length {length} for curve {tag}")
            gmsh.model.mesh.setTransfiniteCurve(tag, divisions + 1)
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2, tag)
        for _, tag in gmsh.model.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(tag)

        faces = classify_outer_faces()
        add_physical(3, matrix, "matrix")
        add_physical(3, core, "core")
        for name, tags in faces.items():
            add_physical(2, tags, name)
        add_physical(0, [find_origin()], "anchor")

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        for axis, (master_name, slave_name) in enumerate(
            (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))
        ):
            gmsh.model.mesh.setPeriodic(
                2, faces[slave_name], faces[master_name], translation(axis)
            )

        expected_type = gmsh.model.mesh.getElementType("Hexahedron", 1, False)
        counts = {}
        for name, volumes in (("matrix", matrix), ("core", core)):
            count = 0
            for volume in volumes:
                types, tags, _ = gmsh.model.mesh.getElements(3, volume)
                present = {int(kind): len(values) for kind, values in zip(types, tags)}
                if set(present) != {expected_type}:
                    raise RuntimeError(f"non-Hex8 elements in {name}: {present}")
                count += present[expected_type]
            counts[name] = count
        core_count = (elements_per_edge // 2) ** 3
        expected_counts = {
            "matrix": elements_per_edge**3 - core_count,
            "core": core_count,
        }
        if counts != expected_counts:
            raise RuntimeError(f"unexpected element counts: {counts}")
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        expected_nodes = (elements_per_edge + 1) ** 3
        if len(node_tags) != expected_nodes:
            raise RuntimeError(
                f"expected {expected_nodes} nodes, got {len(node_tags)}"
            )
        for name, tags in faces.items():
            if name.endswith("max"):
                for tag in tags:
                    master, slaves, masters, _ = gmsh.model.mesh.getPeriodicNodes(2, tag)
                    if int(master) == 0 or len(slaves) == 0 or len(slaves) != len(masters):
                        raise RuntimeError(f"invalid periodic map on {name} patch {tag}")
        gmsh.write(output)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
