"""Run the neo-Hookean regression and featured acceptance decks."""
from __future__ import annotations

import argparse
import copy
import h5py
import json
from pathlib import Path

import numpy as np

from fe_solver.assembly import assemble_internal
from fe_solver.config import Deck, load_deck
from fe_solver.output_fields import unpack_symmetric
from fe_solver.solver import AnalysisResult, run_analysis
from verification.check_j2_necking import extract_history


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECKS = [
    "case_a_hex8.toml", "case_a_hex8_fbar.toml", "case_a_hex20.toml",
    "case_b_hex8.toml", "case_b_hex8_fbar.toml",
    "frame_objectivity_hex8.toml",
    "periodic_core_isochoric_hex8_fbar.toml", "periodic_core_shear_hex8_fbar.toml",
]
AVAILABLE_DECKS = [
    *DEFAULT_DECKS,
    "j2_necking_bar_hex8_fbar.toml",
    "j2_necking_prism_small_hex8.toml",
    "j2_necking_prism_small_hex8_fbar.toml",
]
J2_NECKING_DECKS = {
    "j2_necking_bar_hex8_fbar.toml",
    "j2_necking_prism_small_hex8.toml",
    "j2_necking_prism_small_hex8_fbar.toml",
}


def run_deck(name: str, output_root: Path) -> AnalysisResult:
    source = load_deck(ROOT / "examples" / name)
    data = copy.deepcopy(source.data)
    data["output"]["directory"] = str(output_root / Path(name).stem)
    return run_analysis(Deck(source.path, data, source.curves))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", action="append", choices=AVAILABLE_DECKS)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/fe_solver_acceptance"))
    args = parser.parse_args()
    names = args.deck or DEFAULT_DECKS
    output_root = args.output_root.resolve()
    results: dict[str, AnalysisResult] = {}
    summary: dict[str, dict] = {}
    for name in names:
        result = run_deck(name, output_root)
        results[name] = result
        summary[name] = {
            "increments": len(result.increments),
            "maximum_newton_iterations": max(item.newton_iterations for item in result.increments),
            "total_cutbacks": sum(item.cutbacks for item in result.increments),
            **result.verification,
        }
        if name not in J2_NECKING_DECKS and summary[name]["total_cutbacks"] != 0:
            raise RuntimeError(f"acceptance deck {name} required an unintended cutback")

    if "j2_necking_bar_hex8_fbar.toml" in results:
        history = extract_history(output_root / "j2_necking_bar_hex8_fbar/run.h5")
        elongation = np.asarray(history["end_elongation"])
        plastic_strain = np.asarray(history["maximum_equivalent_plastic_strain"])
        if abs(elongation[-1] - 7.0) > 1.0e-10:
            raise RuntimeError("J2 necking benchmark did not reach 7 mm end elongation")
        if plastic_strain[-1] <= 0.0 or np.any(np.diff(plastic_strain) < -1.0e-12):
            raise RuntimeError("J2 necking benchmark has invalid accumulated plastic strain")
        summary["j2_necking_history"] = history

    if "frame_objectivity_hex8.toml" in results:
        path = output_root / "frame_objectivity_hex8/run.h5"
        with h5py.File(path, "r") as archive:
            times = np.asarray(archive["results/time"])
            stretch = int(np.flatnonzero(np.isclose(times, 0.1))[0])
            final = int(np.flatnonzero(np.isclose(times, 1.0))[0])
            group = archive["results/blocks/0000"]
            rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

            def rotate(values: np.ndarray) -> np.ndarray:
                return np.einsum("ia,eab,jb->eij", rotation, values, rotation)

            stress_stretch = unpack_symmetric(group["cauchy_stress"][stretch])
            stress_final = unpack_symmetric(group["cauchy_stress"][final])
            green_stretch = unpack_symmetric(group["green_lagrange_strain"][stretch])
            green_final = unpack_symmetric(group["green_lagrange_strain"][final])
            almansi_stretch = unpack_symmetric(group["euler_almansi_strain"][stretch])
            almansi_final = unpack_symmetric(group["euler_almansi_strain"][final])
            frame = {
                "cauchy_rotation_error_inf": float(np.max(np.abs(stress_final - rotate(stress_stretch)))),
                "green_invariance_error_inf": float(np.max(np.abs(green_final - green_stretch))),
                "almansi_rotation_error_inf": float(np.max(np.abs(almansi_final - rotate(almansi_stretch)))),
            }
            for step in np.flatnonzero(times >= times[stretch]):
                theta = np.deg2rad(90.0 * (times[step] - 0.1) / 0.9)
                c, s = np.cos(theta), np.sin(theta)
                rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                stress = unpack_symmetric(group["cauchy_stress"][step])
                frame["cauchy_rotation_error_inf"] = max(
                    frame["cauchy_rotation_error_inf"], float(np.max(np.abs(stress - rotate(stress_stretch))))
                )
            frame["initial_transverse_stress_inf"] = float(np.max(np.abs(stress_stretch[:, 1:, 1:])))
            frame["final_sigma11_inf"] = float(np.max(np.abs(stress_final[:, 0, 0])))
            stretch_stress_diagonal = np.mean(np.diagonal(stress_stretch, axis1=1, axis2=2), axis=0)
            final_stress_diagonal = np.mean(np.diagonal(stress_final, axis1=1, axis2=2), axis=0)
            stretch_almansi_diagonal = np.mean(np.diagonal(almansi_stretch, axis1=1, axis2=2), axis=0)
            final_almansi_diagonal = np.mean(np.diagonal(almansi_final, axis1=1, axis2=2), axis=0)
            if max(frame.values()) > 1.0e-10:
                raise RuntimeError(f"frame-objectivity tensor transformation failed: {frame}")
            if not (
                np.argmax(stretch_stress_diagonal) == 0
                and np.argmax(final_stress_diagonal) == 1
                and np.argmax(stretch_almansi_diagonal) == 0
                and np.argmax(final_almansi_diagonal) == 1
            ):
                raise RuntimeError("dominant spatial stress/strain components did not rotate from 11 to 22")
            summary["frame_objectivity"] = frame
    if "case_b_hex8.toml" in results and "case_b_hex8_fbar.toml" in results:
        standard = results["case_b_hex8.toml"]
        fbar = results["case_b_hex8_fbar.toml"]
        standard_assembly = assemble_internal(standard.model, standard.u, standard.u, standard.t, standard.t, False)
        fbar_assembly = assemble_internal(fbar.model, fbar.u, fbar.u, fbar.t, fbar.t, False)
        stress_error = max(
            np.max(np.abs(np.asarray(gs.cauchy_stress) - np.asarray(gf.cauchy_stress)))
            for block_s, block_f in zip(standard_assembly.gauss_output, fbar_assembly.gauss_output)
            for gs, gf in zip(block_s, block_f)
        )
        reaction_s = -standard.constraints.C.T @ standard.lambdas
        reaction_f = -fbar.constraints.C.T @ fbar.lambdas
        comparison = {
            "displacement_error_inf": float(np.max(np.abs(standard.u - fbar.u))),
            "stress_error_inf": float(stress_error),
            "reaction_error_inf": float(np.max(np.abs(reaction_s - reaction_f))),
        }
        if max(comparison.values()) > 1.0e-9:
            raise RuntimeError(f"homogeneous Hex8/F-bar comparison failed: {comparison}")
        summary["case_b_homogeneous_comparison"] = comparison
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
