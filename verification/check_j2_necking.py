"""Extract system-level observables from the J2 circular-bar benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REFERENCE_RADIAL_DISPLACEMENT = -3.740


def extract_history(database: str | Path) -> dict[str, Any]:
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
    final_is_reference_event = bool(
        len(times)
        and abs(float(times[-1]) - 1.0) <= 1.0e-12
        and abs(float(np.mean(u[-1, end_nodes, 2])) - 7.0) <= 1.0e-10
    )
    reference_error = (
        abs(float(u[-1, monitor, 0]) - REFERENCE_RADIAL_DISPLACEMENT)
        / abs(REFERENCE_RADIAL_DISPLACEMENT)
        if final_is_reference_event
        else None
    )
    return {
        "complete_states": int(len(times)),
        "final_time": float(times[-1]),
        "time": times.tolist(),
        "end_elongation": np.mean(u[:, end_nodes, 2], axis=1).tolist(),
        "end_reaction_z": np.sum(reaction[:, end_nodes, 2], axis=1).tolist(),
        "middle_radius": middle_radius.tolist(),
        "monitor_radial_displacement": u[:, monitor, 0].tolist(),
        "maximum_equivalent_plastic_strain": np.max(ep, axis=tuple(range(1, ep.ndim))).tolist(),
        "reference_final_radial_displacement": REFERENCE_RADIAL_DISPLACEMENT,
        "final_relative_reference_error": reference_error,
    }


def plot_history(history: dict[str, Any], output: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    elongation = np.asarray(history["end_elongation"], dtype=float)
    reaction = np.asarray(history["end_reaction_z"], dtype=float)
    displacement = np.asarray(history["monitor_radial_displacement"], dtype=float)
    plastic_strain = np.asarray(history["maximum_equivalent_plastic_strain"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    axes[0].plot(elongation, np.abs(reaction) / 1000.0)
    axes[0].set(xlabel="prescribed end elongation [mm]", ylabel="absolute end reaction [kN]")
    axes[1].plot(elongation, displacement, label="Hex8-Fbar")
    axes[1].plot(
        [7.0], [history["reference_final_radial_displacement"]], "o",
        label="published final reference",
    )
    axes[1].set(
        xlabel="prescribed end elongation [mm]",
        ylabel="middle surface radial displacement [mm]",
    )
    axes[1].legend()
    axes[2].plot(elongation, plastic_strain)
    axes[2].set(
        xlabel="prescribed end elongation [mm]",
        ylabel="maximum equivalent plastic strain",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("J2 circular-bar necking benchmark")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args(argv)
    history = extract_history(args.database)
    if args.require_reference:
        error = history["final_relative_reference_error"]
        if error is None:
            raise SystemExit("database has not reached the 7 mm reference event")
        if error > args.relative_tolerance:
            raise SystemExit(
                "final neck displacement differs from the reference by "
                f"{100.0 * error:.3g}%"
            )
    print(json.dumps(history, indent=2))
    if args.plot is not None:
        plot_history(history, args.plot)


if __name__ == "__main__":
    main()
