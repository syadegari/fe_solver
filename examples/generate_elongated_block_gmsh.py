"""Generate and validate the Case-A structured hexahedral Gmsh mesh.

Requires the official `gmsh` Python package at runtime.
Examples:
  python generate_elongated_block_gmsh.py --element hex8 --out elongated_hex8.msh
  python generate_elongated_block_gmsh.py --element hex20 --out elongated_hex20.msh
"""
from __future__ import annotations

import argparse
import gmsh


def near(a: float, b: float, tol: float = 1.0e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def classify_box_surfaces(Lx: float, Ly: float, Lz: float) -> dict[str, list[int]]:
    groups = {k: [] for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if near(xmin, 0.0) and near(xmax, 0.0):
            groups["xmin"].append(tag)
        elif near(xmin, Lx) and near(xmax, Lx):
            groups["xmax"].append(tag)
        elif near(ymin, 0.0) and near(ymax, 0.0):
            groups["ymin"].append(tag)
        elif near(ymin, Ly) and near(ymax, Ly):
            groups["ymax"].append(tag)
        elif near(zmin, 0.0) and near(zmax, 0.0):
            groups["zmin"].append(tag)
        elif near(zmin, Lz) and near(zmax, Lz):
            groups["zmax"].append(tag)
    if any(len(v) != 1 for v in groups.values()):
        raise RuntimeError(f"Could not classify six box surfaces uniquely: {groups}")
    return groups


def set_structured_hex_constraints(volume: int, nx: int, ny: int, nz: int) -> None:
    for _, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if abs(dx) > abs(dy) + abs(dz):
            nnode = nx + 1
        elif abs(dy) > abs(dx) + abs(dz):
            nnode = ny + 1
        elif abs(dz) > abs(dx) + abs(dy):
            nnode = nz + 1
        else:
            raise RuntimeError(f"Unexpected non-axis-aligned box edge {tag}")
        gmsh.model.mesh.setTransfiniteCurve(tag, nnode)

    for _, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)
    gmsh.model.mesh.setTransfiniteVolume(volume)


def add_physical(dim: int, tags: list[int], name: str) -> None:
    ptag = gmsh.model.addPhysicalGroup(dim, tags)
    gmsh.model.setPhysicalName(dim, ptag, name)


def physical_names() -> set[str]:
    return {
        gmsh.model.getPhysicalName(dim, tag)
        for dim, tag in gmsh.model.getPhysicalGroups()
        if gmsh.model.getPhysicalName(dim, tag)
    }


def validate_mesh(volume: int, element: str, nx: int, ny: int, nz: int) -> None:
    if element == "hex8":
        expected_type = gmsh.model.mesh.getElementType("Hexahedron", 1, False)
        expected_nodes = 8
    elif element == "hex20":
        expected_type = gmsh.model.mesh.getElementType("Hexahedron", 2, True)
        expected_nodes = 20
    else:
        raise ValueError(f"unsupported element: {element}")

    types, elem_tags, _ = gmsh.model.mesh.getElements(3, volume)
    present = {int(t): len(tags) for t, tags in zip(types, elem_tags)}
    expected_count = nx * ny * nz
    if set(present) != {expected_type} or present[expected_type] != expected_count:
        raise RuntimeError(
            f"Unexpected solid elements: got {present}, expected type {expected_type} "
            f"with {expected_count} elements"
        )
    name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(expected_type)
    if dim != 3 or num_nodes != expected_nodes:
        raise RuntimeError(
            f"Unexpected Gmsh properties for {element}: {name=}, {dim=}, {order=}, {num_nodes=}"
        )

    required = {"solid", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    missing = required - physical_names()
    if missing:
        raise RuntimeError(f"Missing required Physical Groups: {sorted(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--element", choices=("hex8", "hex20"), default="hex8")
    ap.add_argument("--out", default="elongated_block.msh")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--nz", type=int, default=2)
    args = ap.parse_args()
    if min(args.nx, args.ny, args.nz) < 1:
        raise ValueError("nx, ny and nz must be positive")

    Lx, Ly, Lz = 4.0, 1.0, 1.0
    gmsh.initialize()
    try:
        gmsh.model.add("elongated_block")
        vol = gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
        gmsh.model.occ.synchronize()
        faces = classify_box_surfaces(Lx, Ly, Lz)
        set_structured_hex_constraints(vol, args.nx, args.ny, args.nz)

        add_physical(3, [vol], "solid")
        for name, tags in faces.items():
            add_physical(2, tags, name)

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)

        if args.element == "hex20":
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
            gmsh.model.mesh.setOrder(2)

        validate_mesh(vol, args.element, args.nx, args.ny, args.nz)
        gmsh.write(args.out)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
