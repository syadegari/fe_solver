"""Run one orthogonal case from the small-prism J2 formulation study."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

from fe_solver.config import Deck, load_deck
from fe_solver.solver import run_analysis
from fe_solver.types import ModelError


ROOT = Path(__file__).resolve().parents[1]
BASE_DECK = ROOT / "examples/j2_necking_prism_small_hex8_fbar.toml"
CIRCULAR_BASE_DECK = ROOT / "examples/j2_necking_bar_hex8_fbar.toml"
REFERENCE_BULK_MODULUS = 164210.0
SOFT_BULK_MODULUS = 0.5 * REFERENCE_BULK_MODULUS


@dataclass(frozen=True)
class StudyCase:
    name: str
    mesh_file: str
    formulation: str
    bulk_modulus: float
    output_directory: str
    purpose: str
    base_geometry: str = "square_prism"


CASES = {
    case.name: case
    for case in (
        StudyCase(
            "coarse_hex8_fbar", "necking_prism_small_hex8.msh", "hex8_fbar",
            REFERENCE_BULK_MODULUS, "results/j2_necking_prism_small_hex8_fbar",
            "baseline anti-locking result",
        ),
        StudyCase(
            "coarse_hex8", "necking_prism_small_hex8.msh", "hex8",
            REFERENCE_BULK_MODULUS, "results/j2_necking_prism_small_hex8",
            "baseline unstabilized control",
        ),
        StudyCase(
            "refined_hex8_fbar", "necking_prism_refined_hex8.msh", "hex8_fbar",
            REFERENCE_BULK_MODULUS,
            "results/j2_formulation_study/refined_hex8_fbar",
            "paired 4x4x48 mesh refinement",
        ),
        StudyCase(
            "refined_hex8", "necking_prism_refined_hex8.msh", "hex8",
            REFERENCE_BULK_MODULUS,
            "results/j2_formulation_study/refined_hex8",
            "paired 4x4x48 mesh refinement",
        ),
        StudyCase(
            "soft_bulk_hex8_fbar", "necking_prism_small_hex8.msh", "hex8_fbar",
            SOFT_BULK_MODULUS,
            "results/j2_formulation_study/soft_bulk_hex8_fbar",
            "half-bulk-modulus material sensitivity",
        ),
        StudyCase(
            "soft_bulk_hex8", "necking_prism_small_hex8.msh", "hex8",
            SOFT_BULK_MODULUS,
            "results/j2_formulation_study/soft_bulk_hex8",
            "half-bulk-modulus material sensitivity",
        ),
        StudyCase(
            "coarse_hex20", "necking_prism_small_hex20.msh", "hex20",
            REFERENCE_BULK_MODULUS,
            "results/j2_formulation_study/coarse_hex20",
            "higher-order standard-element reference",
        ),
        StudyCase(
            "circular_hex8_fbar", "necking_bar_quarter_hex8.msh", "hex8_fbar",
            REFERENCE_BULK_MODULUS, "results/j2_necking_bar_hex8_fbar",
            "960-element circular benchmark baseline", "circular_bar",
        ),
        StudyCase(
            "refined_circular_hex8_fbar", "necking_bar_quarter_refined_hex8.msh",
            "hex8_fbar", REFERENCE_BULK_MODULUS,
            "results/j2_formulation_study/refined_circular_hex8_fbar",
            "7680-element circular benchmark refinement", "circular_bar",
        ),
    )
}


def elastic_poisson_ratio(shear_modulus: float, bulk_modulus: float) -> float:
    return (3.0 * bulk_modulus - 2.0 * shear_modulus) / (
        2.0 * (3.0 * bulk_modulus + shear_modulus)
    )


def build_case_deck(
    case_name: str,
    *,
    restart_from: str = "",
    output_directory: str | Path | None = None,
    grow_if_newton_iterations_le: int | None = None,
) -> Deck:
    try:
        case = CASES[case_name]
    except KeyError as exc:
        raise ValueError(f"unknown J2 formulation-study case {case_name!r}") from exc
    base_path = CIRCULAR_BASE_DECK if case.base_geometry == "circular_bar" else BASE_DECK
    base = load_deck(base_path)
    data = copy.deepcopy(base.data)
    data["analysis"]["name"] = f"j2_study_{case.name}"
    data["mesh"]["file"] = case.mesh_file
    data["materials"][0]["properties"]["bulk_modulus"] = case.bulk_modulus
    data["element_assignments"][0]["formulation"] = case.formulation
    data["output"]["directory"] = (
        str(Path(output_directory).resolve())
        if output_directory is not None
        else case.output_directory
    )
    if grow_if_newton_iterations_le is not None:
        if grow_if_newton_iterations_le < 0:
            raise ValueError("Newton-iteration growth threshold must be nonnegative")
        data["time"]["grow_if_newton_iterations_le"] = (
            grow_if_newton_iterations_le
        )
    data["restart"]["restart_from"] = restart_from
    return Deck(base.path, data, base.curves)


def describe_cases() -> str:
    mu = float(load_deck(BASE_DECK).data["materials"][0]["properties"]["shear_modulus"])
    name_width = max(map(len, CASES))
    lines = []
    for case in CASES.values():
        nu = elastic_poisson_ratio(mu, case.bulk_modulus)
        lines.append(
            f"{case.name:{name_width}s} {case.formulation:9s} "
            f"K={case.bulk_modulus:g} MPa nu_elastic={nu:.6f}  {case.purpose}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="run one case from the small-prism J2 formulation study"
    )
    parser.add_argument("--case", choices=tuple(CASES), default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--stop-time", type=float, default=None)
    parser.add_argument(
        "--restart-from", default="",
        help="restart file, resolved relative to the selected base deck",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=None,
        help="override the case output directory (recommended for study batches)",
    )
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument(
        "--grow-if-newton-iterations-le",
        type=int,
        default=None,
        help="override the deck's accepted-step growth threshold for this run",
    )
    args = parser.parse_args(argv)
    if args.list_cases:
        print(describe_cases())
        return
    if args.case is None:
        parser.error("--case is required unless --list-cases is used")
    if (
        args.grow_if_newton_iterations_le is not None
        and args.grow_if_newton_iterations_le < 0
    ):
        parser.error("--grow-if-newton-iterations-le must be nonnegative")
    try:
        result = run_analysis(
            build_case_deck(
                args.case,
                restart_from=args.restart_from,
                output_directory=args.output_directory,
                grow_if_newton_iterations_le=args.grow_if_newton_iterations_le,
            ),
            stop_time=args.stop_time,
            num_processes=args.num_processes,
            debug_timing=args.debug_timing,
        )
    except ModelError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"converged {len(result.increments)} increments to t={result.t:.12g}; "
        f"max|u|={abs(result.u).max():.6g}; "
        f"force balance={result.verification['force_balance_inf']:.3e}"
    )


if __name__ == "__main__":
    main()
