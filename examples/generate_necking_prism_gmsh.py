"""Generate a small tapered square-prism diagnostic for J2 necking."""
from __future__ import annotations

import argparse
from pathlib import Path

import gmsh
import numpy as np


RADIUS = 6.413
LENGTH = 53.334
END_HALF_WIDTH = RADIUS * np.sqrt(np.pi) / 2.0


def near(a: float, b: float, tolerance: float = 1.0e-6) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def add_physical(dimension: int, entities: list[int], name: str) -> None:
    tag = gmsh.model.addPhysicalGroup(dimension, entities)
    gmsh.model.setPhysicalName(dimension, tag, name)


def normalize_msh_whitespace(path: str) -> None:
    output = Path(path)
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def classify_unit_box() -> tuple[dict[str, list[int]], int]:
    surfaces = {
        name: []
        for name in (
            "symmetry_x", "symmetry_y", "midplane", "loaded_end", "outer_surface"
        )
    }
    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if near(xmin, 0.0) and near(xmax, 0.0):
            surfaces["symmetry_x"].append(tag)
        elif near(ymin, 0.0) and near(ymax, 0.0):
            surfaces["symmetry_y"].append(tag)
        elif near(zmin, 0.0) and near(zmax, 0.0):
            surfaces["midplane"].append(tag)
        elif near(zmin, 1.0) and near(zmax, 1.0):
            surfaces["loaded_end"].append(tag)
        else:
            surfaces["outer_surface"].append(tag)
    if any(not entities for entities in surfaces.values()):
        raise RuntimeError(f"failed to classify unit-box surfaces: {surfaces}")
    monitor = []
    for _, tag in gmsh.model.getEntities(0):
        box = gmsh.model.getBoundingBox(0, tag)
        if all(near(value, target) for value, target in zip(box, (1, 0, 0, 1, 0, 0))):
            monitor.append(tag)
    if len(monitor) != 1:
        raise RuntimeError(f"failed to identify neck monitor point: {monitor}")
    return surfaces, monitor[0]


def axial_coordinate(parameter: float) -> float:
    if parameter <= 0.5:
        return LENGTH * parameter / 3.0
    return LENGTH / 6.0 + 2.0 * LENGTH * (parameter - 0.5) / 3.0


def map_mesh(imperfection: float) -> None:
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    for tag, (xi, eta, axial_parameter) in zip(tags, coordinates):
        z = axial_coordinate(float(axial_parameter))
        scale = 1.0 - imperfection + 2.0 * imperfection * z / LENGTH
        half_width = END_HALF_WIDTH * scale
        gmsh.model.mesh.setNode(
            int(tag), [half_width * float(xi), half_width * float(eta), z], []
        )


def validate(volume: int, nx: int, ny: int, nz: int) -> None:
    hex8 = gmsh.model.mesh.getElementType("Hexahedron", 1, False)
    types, blocks, _ = gmsh.model.mesh.getElements(3, volume)
    counts = {int(kind): len(tags) for kind, tags in zip(types, blocks)}
    expected = nx * ny * nz
    if counts != {hex8: expected}:
        raise RuntimeError(f"unexpected mesh {counts}; expected {expected} Hex8 elements")
    tags = np.concatenate([np.asarray(block, dtype=np.int64) for block in blocks])
    quality = np.asarray(gmsh.model.mesh.getElementQualities(tags, "minSJ"))
    if quality.min() <= 0.0:
        raise RuntimeError("generated prism has a nonpositive scaled Jacobian")
    print(f"validated {expected} Hex8 elements; minimum scaled Jacobian={quality.min():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="necking_prism_small_hex8.msh")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--imperfection", type=float, default=0.018)
    args = parser.parse_args()
    if min(args.nx, args.ny, args.nz) < 1 or args.nz % 2:
        raise ValueError("division counts must be positive and nz must be even")
    if not 0.0 <= args.imperfection < 1.0:
        raise ValueError("imperfection must lie in [0,1)")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("necking_prism_small")
        volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        surfaces, monitor = classify_unit_box()
        for _, tag in gmsh.model.getEntities(1):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
            axis = int(np.argmax(np.abs([xmax - xmin, ymax - ymin, zmax - zmin])))
            gmsh.model.mesh.setTransfiniteCurve(tag, (args.nx, args.ny, args.nz)[axis] + 1)
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(2, tag)
        gmsh.model.mesh.setTransfiniteVolume(volume)
        add_physical(3, [volume], "solid")
        for name, entities in surfaces.items():
            add_physical(2, entities, name)
        add_physical(0, [monitor], "neck_monitor")

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        map_mesh(args.imperfection)
        validate(volume, args.nx, args.ny, args.nz)
        gmsh.write(args.out)
        normalize_msh_whitespace(args.out)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
