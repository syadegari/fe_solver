"""Extract the J2 formulation-study diagnostics directly from HDF5 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verification.check_j2_necking import (
    extract_history as extract_circular_history,
    load_reference_curves,
)
from verification.check_j2_prism import (
    PrismHistory,
    compare_prism_histories,
    extract_prism_history,
    plot_prism_histories,
)
from verification.run_j2_formulation_study import ROOT


COARSE_DATABASES = {
    "coarse_hex8_fbar": ROOT / "examples/results/j2_necking_prism_small_hex8_fbar/run.h5",
    "coarse_hex8": ROOT / "examples/results/j2_necking_prism_small_hex8/run.h5",
}
RUN_DATABASES = {
    name: Path("results") / name / "run.h5"
    for name in (
        "soft_bulk_hex8_fbar",
        "soft_bulk_hex8",
        "coarse_hex20",
        "refined_hex8_fbar",
        "refined_hex8",
        "circular_hex8_fbar",
        "refined_circular_hex8_fbar",
    )
}
PRISM_COMPARISONS = {
    "reference_bulk_formulation_gap": ("coarse_hex8_fbar", "coarse_hex8"),
    "half_bulk_formulation_gap": ("soft_bulk_hex8_fbar", "soft_bulk_hex8"),
    "fbar_mesh_refinement": ("coarse_hex8_fbar", "refined_hex8_fbar"),
    "hex8_mesh_refinement": ("coarse_hex8", "refined_hex8"),
    "hex20_vs_coarse_fbar": ("coarse_hex8_fbar", "coarse_hex20"),
    "hex20_vs_coarse_hex8": ("coarse_hex8", "coarse_hex20"),
    "fbar_bulk_sensitivity": ("coarse_hex8_fbar", "soft_bulk_hex8_fbar"),
    "hex8_bulk_sensitivity": ("coarse_hex8", "soft_bulk_hex8"),
}


def _require_time(name: str, final_time: float, expected: float, *, at_least: bool) -> None:
    tolerance = 1.0e-12
    valid = final_time >= expected - tolerance if at_least else abs(final_time - expected) <= tolerance
    if not valid:
        relation = "at least" if at_least else "exactly"
        raise RuntimeError(
            f"{name} ends at t={final_time:.12g}; expected {relation} t={expected:g}"
        )


def compare_circular_histories(
    coarse: dict[str, Any], refined: dict[str, Any]
) -> dict[str, Any]:
    coarse_displacement = float(coarse["monitor_radial_displacement"][-1])
    refined_displacement = float(refined["monitor_radial_displacement"][-1])

    def peak(history: dict[str, Any]) -> dict[str, float]:
        reaction = np.abs(np.asarray(history["end_reaction_z"], dtype=float))
        index = int(np.argmax(reaction))
        return {
            "time": float(history["time"][index]),
            "end_elongation": float(history["end_elongation"][index]),
            "absolute_reaction": float(reaction[index]),
        }

    return {
        "coarse_database": coarse["database"],
        "refined_database": refined["database"],
        "common_end_time": min(float(coarse["final_time"]), float(refined["final_time"])),
        "coarse_peak": peak(coarse),
        "refined_peak": peak(refined),
        "coarse_final_radial_displacement": coarse_displacement,
        "refined_final_radial_displacement": refined_displacement,
        "refined_minus_coarse_radial_displacement": (
            refined_displacement - coarse_displacement
        ),
        "coarse_final_relative_reference_error": coarse[
            "final_relative_reference_error"
        ],
        "refined_final_relative_reference_error": refined[
            "final_relative_reference_error"
        ],
        "coarse_final_maximum_equivalent_plastic_strain": float(
            coarse["maximum_equivalent_plastic_strain"][-1]
        ),
        "refined_final_maximum_equivalent_plastic_strain": float(
            refined["maximum_equivalent_plastic_strain"][-1]
        ),
        "coarse_final_maximum_plastic_strain_initial_z": float(
            coarse["maximum_plastic_strain_initial_z"][-1]
        ),
        "refined_final_maximum_plastic_strain_initial_z": float(
            refined["maximum_plastic_strain_initial_z"][-1]
        ),
    }


def plot_circular_histories(
    coarse: dict[str, Any], refined: dict[str, Any], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    for history, label, style in (
        (coarse, "coarse Hex8-Fbar", "-"),
        (refined, "refined Hex8-Fbar", "--"),
    ):
        elongation = np.asarray(history["end_elongation"], dtype=float)
        axes[0].plot(
            elongation,
            np.abs(np.asarray(history["end_reaction_z"], dtype=float)) / 1000.0,
            style,
            label=label,
        )
        axes[1].plot(
            elongation, history["monitor_radial_displacement"], style, label=label
        )
        axes[2].plot(
            elongation,
            history["maximum_equivalent_plastic_strain"],
            style,
            label=label,
        )
    reference = load_reference_curves()
    for name, label, marker in (
        ("simo_hughes", "Simo--Hughes tabulation", "s"),
        ("elguedj_hughes", "Elguedj--Hughes tabulation", "^"),
    ):
        series = reference[name]
        axes[1].plot(
            series["end_elongation_mm"],
            series["radial_displacement_mm"],
            linestyle=":",
            marker=marker,
            markersize=4,
            label=label,
        )
    axes[0].set(xlabel="prescribed end elongation [mm]", ylabel="absolute end reaction [kN]")
    axes[1].set(xlabel="prescribed end elongation [mm]", ylabel="monitored radial displacement [mm]")
    axes[2].set(xlabel="prescribed end elongation [mm]", ylabel="maximum equivalent plastic strain")
    if min(float(coarse["final_time"]), float(refined["final_time"])) >= 1.0 - 1.0e-12:
        axes[1].plot(
            [7.0], [coarse["reference_final_radial_displacement"]], "o",
            label="ANSYS/Simo endpoint",
        )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle("J2 circular-bar mesh refinement")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_report(
    run_root: Path,
    output_directory: Path,
    *,
    expected_time: float,
) -> dict[str, Any]:
    prism: dict[str, PrismHistory] = {}
    for name, database in COARSE_DATABASES.items():
        history = extract_prism_history(database)
        _require_time(name, float(history.time[-1]), expected_time, at_least=True)
        prism[name] = history
    for name in (
        "soft_bulk_hex8_fbar", "soft_bulk_hex8", "coarse_hex20",
        "refined_hex8_fbar", "refined_hex8",
    ):
        database = run_root / RUN_DATABASES[name]
        history = extract_prism_history(database)
        _require_time(name, float(history.time[-1]), expected_time, at_least=False)
        prism[name] = history

    output_directory.mkdir(parents=True, exist_ok=True)
    prism_comparisons = {}
    for comparison_name, (reference_name, candidate_name) in PRISM_COMPARISONS.items():
        reference = prism[reference_name]
        candidate = prism[candidate_name]
        prism_comparisons[comparison_name] = compare_prism_histories(reference, candidate)
        plot_prism_histories(
            reference, candidate, output_directory / f"{comparison_name}.png"
        )

    circular = {}
    for name in ("circular_hex8_fbar", "refined_circular_hex8_fbar"):
        history = extract_circular_history(run_root / RUN_DATABASES[name])
        _require_time(name, float(history["final_time"]), expected_time, at_least=False)
        circular[name] = history
    circular_comparison = compare_circular_histories(
        circular["circular_hex8_fbar"], circular["refined_circular_hex8_fbar"]
    )
    plot_circular_histories(
        circular["circular_hex8_fbar"],
        circular["refined_circular_hex8_fbar"],
        output_directory / "circular_mesh_refinement.png",
    )
    return {
        "expected_time": expected_time,
        "run_root": str(run_root),
        "prism_cases": {name: history.summary() for name, history in prism.items()},
        "prism_comparisons": prism_comparisons,
        "circular_cases": circular,
        "circular_comparison": circular_comparison,
        "circular_reference_curves": load_reference_curves(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-time", type=float, choices=(0.5, 1.0), default=1.0)
    parser.add_argument("--output-directory", type=Path, default=None)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    output = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else run_root / f"report_t{args.expected_time:g}"
    )
    report = build_report(run_root, output, expected_time=args.expected_time)
    report_path = output / "study_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
