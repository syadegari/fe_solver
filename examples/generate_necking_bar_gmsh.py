"""Generate a structured one-eighth model of the circular necking bar.

The quarter cross-section uses three transfinite blocks: a regular central
square and two outer sectors. Unlike a one-block square-to-disk map, no
logical corner is mapped onto the circular boundary at 45 degrees.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import gmsh
import numpy as np


RADIUS = 6.413
LENGTH = 53.334


def near(a: float, b: float, tolerance: float = 1.0e-6) -> bool:
    """Compare geometry bounding-box coordinates including Gmsh tolerance."""
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def add_physical(dimension: int, entities: list[int], name: str) -> None:
    tag = gmsh.model.addPhysicalGroup(dimension, entities)
    gmsh.model.setPhysicalName(dimension, tag, name)


def normalize_msh_whitespace(path: str) -> None:
    output = Path(path)
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def build_three_block_cross_section(
    core_fraction: float, core_divisions: int, radial_divisions: int, nz: int,
) -> tuple[list[int], int]:
    """Create and extrude a unit-radius, unit-length three-block quadrant."""
    geo = gmsh.model.geo
    a = core_fraction
    p0 = geo.addPoint(0.0, 0.0, 0.0)
    p1 = geo.addPoint(a, 0.0, 0.0)
    p2 = geo.addPoint(a, a, 0.0)
    p3 = geo.addPoint(0.0, a, 0.0)
    p4 = geo.addPoint(1.0, 0.0, 0.0)
    p5 = geo.addPoint(1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0)
    p6 = geo.addPoint(0.0, 1.0, 0.0)

    l01 = geo.addLine(p0, p1)
    l12 = geo.addLine(p1, p2)
    l23 = geo.addLine(p2, p3)
    l30 = geo.addLine(p3, p0)
    l14 = geo.addLine(p1, p4)
    arc45 = geo.addCircleArc(p4, p0, p5)
    l52 = geo.addLine(p5, p2)
    arc56 = geo.addCircleArc(p5, p0, p6)
    l63 = geo.addLine(p6, p3)

    surfaces = [
        geo.addPlaneSurface([geo.addCurveLoop([l01, l12, l23, l30])]),
        geo.addPlaneSurface([geo.addCurveLoop([l14, arc45, l52, -l12])]),
        geo.addPlaneSurface([geo.addCurveLoop([-l23, -l52, arc56, l63])]),
    ]
    for curve in (l01, l12, l23, l30, arc45, arc56):
        geo.mesh.setTransfiniteCurve(curve, core_divisions + 1)
    for curve in (l14, l52, l63):
        geo.mesh.setTransfiniteCurve(curve, radial_divisions + 1)
    for surface in surfaces:
        geo.mesh.setTransfiniteSurface(surface)
        geo.mesh.setRecombine(2, surface)

    extruded = geo.extrude(
        [(2, surface) for surface in surfaces],
        0.0, 0.0, 1.0,
        numElements=[nz],
        recombine=True,
    )
    geo.synchronize()
    volumes = [tag for dimension, tag in extruded if dimension == 3]
    if len(volumes) != 3:
        raise RuntimeError(f"expected three extruded volumes, found {volumes}")
    return volumes, p4


def classify_external_surfaces(volumes: list[int]) -> dict[str, list[int]]:
    """Classify exterior faces while omitting shared block interfaces."""
    occurrences: Counter[int] = Counter()
    for volume in volumes:
        for dimension, tag in gmsh.model.getBoundary(
            [(3, volume)], combined=False, oriented=False
        ):
            if dimension == 2:
                occurrences[int(tag)] += 1
    exterior = sorted(tag for tag, count in occurrences.items() if count == 1)
    surfaces = {
        name: []
        for name in (
            "symmetry_x", "symmetry_y", "midplane", "loaded_end", "outer_surface"
        )
    }
    for tag in exterior:
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
    if len(exterior) != 12 or any(not entities for entities in surfaces.values()):
        raise RuntimeError(
            f"failed to classify the {len(exterior)} exterior surfaces: {surfaces}"
        )
    return surfaces


def axial_coordinate(parameter: float) -> float:
    """Put half the axial layers in the central third of the half-model."""
    if parameter <= 0.5:
        return LENGTH * parameter / 3.0
    return LENGTH / 6.0 + 2.0 * LENGTH * (parameter - 0.5) / 3.0


def map_mesh_to_tapered_quarter_cylinder() -> None:
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    for tag, (xi, eta, axial_parameter) in zip(tags, coordinates):
        z = axial_coordinate(float(axial_parameter))
        radius = RADIUS * (0.982 + 0.036 * z / LENGTH)
        gmsh.model.mesh.setNode(
            int(tag), [radius * float(xi), radius * float(eta), z], []
        )


def validate(
    volumes: list[int], core_divisions: int, radial_divisions: int, nz: int,
) -> None:
    hex8 = gmsh.model.mesh.getElementType("Hexahedron", 1, False)
    element_tags: list[int] = []
    types_found: set[int] = set()
    for volume in volumes:
        types, tag_blocks, _ = gmsh.model.mesh.getElements(3, volume)
        for kind, tags in zip(types, tag_blocks):
            types_found.add(int(kind))
            element_tags.extend(map(int, tags))
    expected = (core_divisions**2 + 2 * core_divisions * radial_divisions) * nz
    if types_found != {hex8} or len(element_tags) != expected:
        raise RuntimeError(
            f"unexpected volume mesh: types {types_found}, {len(element_tags)} elements; "
            f"expected {expected} Hex8 elements"
        )
    quality = np.asarray(
        gmsh.model.mesh.getElementQualities(element_tags, "minSJ"), dtype=float
    )
    if len(quality) != expected or not np.all(np.isfinite(quality)) or quality.min() <= 0.0:
        raise RuntimeError("generated mesh has a nonpositive or invalid scaled Jacobian")
    required = {
        "solid", "symmetry_x", "symmetry_y", "midplane", "loaded_end",
        "outer_surface", "neck_monitor",
    }
    names = {
        gmsh.model.getPhysicalName(dim, tag)
        for dim, tag in gmsh.model.getPhysicalGroups()
    }
    if required - names:
        raise RuntimeError(f"missing Physical Groups: {sorted(required - names)}")
    print(
        f"validated {expected} Hex8 elements; minimum scaled Jacobian="
        f"{quality.min():.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="necking_bar_quarter_hex8.msh")
    parser.add_argument("--core-divisions", type=int, default=4)
    parser.add_argument("--radial-divisions", type=int, default=3)
    parser.add_argument("--core-fraction", type=float, default=0.4)
    parser.add_argument("--nz", type=int, default=24)
    args = parser.parse_args()
    if min(args.core_divisions, args.radial_divisions, args.nz) < 2:
        raise ValueError("all division counts must be at least two")
    if args.nz % 2:
        raise ValueError("nz must be even")
    if not 0.0 < args.core_fraction < 1.0 / np.sqrt(2.0):
        raise ValueError("core_fraction must lie between zero and 1/sqrt(2)")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("necking_bar_quarter")
        volumes, monitor = build_three_block_cross_section(
            args.core_fraction, args.core_divisions, args.radial_divisions, args.nz
        )
        surfaces = classify_external_surfaces(volumes)
        add_physical(3, volumes, "solid")
        for name, entities in surfaces.items():
            add_physical(2, entities, name)
        add_physical(0, [monitor], "neck_monitor")

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        map_mesh_to_tapered_quarter_cylinder()
        validate(volumes, args.core_divisions, args.radial_divisions, args.nz)
        gmsh.write(args.out)
        normalize_msh_whitespace(args.out)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
