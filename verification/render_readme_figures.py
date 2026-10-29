"""Render the README necking comparison from retained histories; no FE updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"


def export_necking_history(report: Path, target: Path) -> None:
    """Retain only plotted histories/reference data, without machine paths."""
    study = json.loads(report.read_text())
    cases = {}
    for name, label in (("circular_hex8_fbar", "Hex8-Fbar: 960 elements"),
                        ("refined_circular_hex8_fbar", "Hex8-Fbar: 7680 elements")):
        source = study["circular_cases"][name]
        cases[name] = {
            "label": label,
            "end_elongation_mm": source["end_elongation"],
            "radial_displacement_mm": source["monitor_radial_displacement"],
        }
    data = {
        "description": "Completed circular-bar study; plotted monitor-point displacement.",
        "study_record": "verification/j2_formulation_study/FROZEN_STATE.md",
        "reference_record": "verification/j2_formulation_study/REFERENCE_DATA.md",
        "cases": cases,
        "references": study["circular_reference_curves"],
    }
    target.write_text(json.dumps(data, indent=2) + "\n")


def plot_necking(history: Path, output: Path) -> None:
    data = json.loads(history.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for case, color in zip(data["cases"].values(), ("#176b9b", "#d27621")):
        ax.plot(case["end_elongation_mm"], case["radial_displacement_mm"],
                label=case["label"], color=color, linewidth=2)
    references = data["references"]
    for name, marker, label in (("simo_hughes", "o", "Simo & Hughes"),
                                ("elguedj_hughes", "s", "Elguedj & Hughes")):
        series = references[name]
        ax.plot(series["end_elongation_mm"], series["radial_displacement_mm"],
                marker, markerfacecolor="none", color="#343a40", markersize=6,
                label=label)
    endpoint = references["ansys_simo_1992_endpoint"]
    ax.plot(endpoint["end_elongation_mm"], endpoint["radial_displacement_mm"],
            "x", color="#343a40", markersize=8, label="ANSYS VM318 endpoint")
    ax.set(title="Circular-bar necking", xlabel="Prescribed end elongation [mm]",
           ylabel="Middle-surface radial displacement [mm]", xlim=(0, 7.12))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--necking-report", type=Path,
                        help="optional completed study_summary.json to refresh portable history")
    parser.add_argument("--output", type=Path, default=ASSETS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    history = args.output / "necking_history.json"
    if args.necking_report:
        export_necking_history(args.necking_report, history)
    plot_necking(history, args.output / "necking_response.png")


if __name__ == "__main__":
    main()
