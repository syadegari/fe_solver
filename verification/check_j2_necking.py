"""Extract system-level observables from the J2 circular-bar benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REFERENCE_RADIAL_DISPLACEMENT = -3.740


def extract_history(database: str | Path) -> dict[str, list[float] | float]:
    with h5py.File(Path(database), "r") as archive:
        complete = int(archive["results"].attrs["n_complete_steps"])
        times = np.asarray(archive["results/time"][:complete], dtype=float)
        X = np.asarray(archive["mesh/reference_coordinates"], dtype=float)
        u = np.asarray(archive["results/nodal/displacement"][:complete], dtype=float)
        reaction = np.asarray(
            archive["results/nodal/constraint_reaction"][:complete], dtype=float
        )
        state = archive["results/blocks/0000/state"]
        if "equivalent_plastic_strain" not in state:
            raise RuntimeError("benchmark database has no equivalent_plastic_strain field")
        ep = np.asarray(state["equivalent_plastic_strain"][:complete], dtype=float)

    scale = max(float(np.max(np.abs(X))), 1.0)
    tolerance = 1.0e-10 * scale
    middle_nodes = np.flatnonzero(np.abs(X[:, 2] - np.min(X[:, 2])) <= tolerance)
    end_nodes = np.flatnonzero(np.abs(X[:, 2] - np.max(X[:, 2])) <= tolerance)
    monitor = middle_nodes[
        np.argmax(X[middle_nodes, 0] - 1.0e3 * np.abs(X[middle_nodes, 1]))
    ]
    current = X[None, :, :] + u
    # The axis itself has zero radius; monitor the outer middle-section nodes.
    initial_middle_radius = np.sqrt(X[middle_nodes, 0] ** 2 + X[middle_nodes, 1] ** 2)
    surface = middle_nodes[np.isclose(initial_middle_radius, initial_middle_radius.max(), rtol=0, atol=tolerance)]
    middle_radius = np.mean(
        np.sqrt(current[:, surface, 0] ** 2 + current[:, surface, 1] ** 2), axis=1
    )
    return {
        "time": times.tolist(),
        "end_elongation": np.mean(u[:, end_nodes, 2], axis=1).tolist(),
        "end_reaction_z": np.sum(reaction[:, end_nodes, 2], axis=1).tolist(),
        "middle_radius": middle_radius.tolist(),
        "monitor_radial_displacement": u[:, monitor, 0].tolist(),
        "maximum_equivalent_plastic_strain": np.max(ep, axis=tuple(range(1, ep.ndim))).tolist(),
        "reference_final_radial_displacement": REFERENCE_RADIAL_DISPLACEMENT,
        "final_relative_reference_error": abs(
            float(u[-1, monitor, 0]) - REFERENCE_RADIAL_DISPLACEMENT
        ) / abs(REFERENCE_RADIAL_DISPLACEMENT),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    history = extract_history(args.database)
    if args.require_reference and history["final_relative_reference_error"] > args.relative_tolerance:
        raise SystemExit(
            "final neck displacement differs from the reference by "
            f"{100.0 * history['final_relative_reference_error']:.3g}%"
        )
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
