"""Generate and validate the Case-B 8x8x8 periodic linear-Hex8 cube.

Requires the official `gmsh` Python package at runtime. The +x/+y/+z surfaces
are slave copies of the corresponding minus surfaces through pure translation.
"""
from __future__ import annotations

import argparse
import gmsh


def near(a: float, b: float, tol: float = 1.0e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def classify_surfaces(L: float) -> dict[str, int]:
    groups: dict[str, int] = {}
    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if near(xmin, 0.0) and near(xmax, 0.0):
            groups["xmin"] = tag
        elif near(xmin, L) and near(xmax, L):
            groups["xmax"] = tag
        elif near(ymin, 0.0) and near(ymax, 0.0):
            groups["ymin"] = tag
        elif near(ymin, L) and near(ymax, L):
            groups["ymax"] = tag
        elif near(zmin, 0.0) and near(zmax, 0.0):
            groups["zmin"] = tag
        elif near(zmin, L) and near(zmax, L):
            groups["zmax"] = tag
    expected = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    if set(groups) != expected:
        raise RuntimeError(f"Could not classify box surfaces: {groups}")
    return groups


def find_origin_point() -> int:
    candidates = []
    for _, tag in gmsh.model.getEntities(0):
        x0, y0, z0, _, _, _ = gmsh.model.getBoundingBox(0, tag)
        if near(x0, 0.0) and near(y0, 0.0) and near(z0, 0.0):
            candidates.append(tag)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one origin point, got {candidates}")
    return candidates[0]


def add_physical(dim: int, tags: list[int], name: str) -> None:
    ptag = gmsh.model.addPhysicalGroup(dim, tags)
    gmsh.model.setPhysicalName(dim, ptag, name)


def translation(dx: float, dy: float, dz: float) -> list[float]:
    return [
        1.0, 0.0, 0.0, dx,
        0.0, 1.0, 0.0, dy,
        0.0, 0.0, 1.0, dz,
        0.0, 0.0, 0.0, 1.0,
    ]


def physical_names() -> set[str]:
    return {
        gmsh.model.getPhysicalName(dim, tag)
        for dim, tag in gmsh.model.getPhysicalGroups()
        if gmsh.model.getPhysicalName(dim, tag)
    }


def validate_periodic_map(slave: int, master: int, delta: tuple[float, float, float]) -> None:
    tag_master, slave_nodes, master_nodes, _ = gmsh.model.mesh.getPeriodicNodes(
        2, slave, includeHighOrderNodes=False
    )
    if int(tag_master) != int(master):
        raise RuntimeError(f"Periodic master mismatch for slave surface {slave}: {tag_master} != {master}")
    if len(slave_nodes) == 0 or len(slave_nodes) != len(master_nodes):
        raise RuntimeError(f"Invalid periodic node map for slave surface {slave}")

    dx, dy, dz = delta
    for sn, mn in zip(slave_nodes, master_nodes):
        xs = gmsh.model.mesh.getNode(int(sn))[0]
        xm = gmsh.model.mesh.getNode(int(mn))[0]
        expected = (xm[0] + dx, xm[1] + dy, xm[2] + dz)
        if not all(near(float(a), float(b), 1.0e-8) for a, b in zip(xs, expected)):
            raise RuntimeError(
                f"Periodic pair {int(sn)}->{int(mn)} is inconsistent with translation {delta}: "
                f"slave={xs}, expected={expected}"
            )


def validate_mesh(volume: int, faces: dict[str, int], n: int, L: float) -> None:
    expected_type = gmsh.model.mesh.getElementType("Hexahedron", 1, False)
    types, elem_tags, _ = gmsh.model.mesh.getElements(3, volume)
    present = {int(t): len(tags) for t, tags in zip(types, elem_tags)}
    expected_count = n ** 3
    if set(present) != {expected_type} or present[expected_type] != expected_count:
        raise RuntimeError(
            f"Unexpected solid elements: got {present}, expected type {expected_type} with {expected_count} elements"
        )

    required = {"solid", "anchor", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    missing = required - physical_names()
    if missing:
        raise RuntimeError(f"Missing required Physical Groups: {sorted(missing)}")

    validate_periodic_map(faces["xmax"], faces["xmin"], (L, 0.0, 0.0))
    validate_periodic_map(faces["ymax"], faces["ymin"], (0.0, L, 0.0))
    validate_periodic_map(faces["zmax"], faces["zmin"], (0.0, 0.0, L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="periodic_cube_8x8x8.msh")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    if args.n < 1:
        raise ValueError("n must be positive")

    L = 1.0
    gmsh.initialize()
    try:
        gmsh.model.add("periodic_cube")
        vol = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, L, L)
        gmsh.model.occ.synchronize()
        faces = classify_surfaces(L)
        origin = find_origin_point()

        for _, tag in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(tag, args.n + 1)
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2, tag)
        gmsh.model.mesh.setTransfiniteVolume(vol)

        gmsh.model.mesh.setPeriodic(2, [faces["xmax"]], [faces["xmin"]], translation(L, 0.0, 0.0))
        gmsh.model.mesh.setPeriodic(2, [faces["ymax"]], [faces["ymin"]], translation(0.0, L, 0.0))
        gmsh.model.mesh.setPeriodic(2, [faces["zmax"]], [faces["zmin"]], translation(0.0, 0.0, L))

        add_physical(3, [vol], "solid")
        for name, tag in faces.items():
            add_physical(2, [tag], name)
        add_physical(0, [origin], "anchor")

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        validate_mesh(vol, faces, args.n, L)
        gmsh.write(args.out)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
