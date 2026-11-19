"""Run the frozen patch-test cases as simulations and print the evidence.

Each case is solved through the full deck -> KKT -> Newton path from its
committed inputs, results are kept under the case's ``results/`` directory for
ParaView, and the measured quantities behind the patch-test claim are printed
with their thresholds:

    conda run -n py3.14 python -m verification.patch_tests.verify [hex8 ...]

Exit status is nonzero if any measurement exceeds its threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fe_solver.assembly import assemble_internal
from fe_solver.preprocess import prepare_analysis
from fe_solver.solver import run_analysis
from verification.patch_tests.patch_cube import (
    CASES, F_FINITE, NEO_KAPPA, analytic_internal_force, check_frozen_case,
    interior_dof_mask, neo_stress,
)


ROOT = Path(__file__).resolve().parent

THRESHOLDS = {
    "interior_residual_rel": 1.0e-11,
    "F_error": 1.0e-12,
    "J_error": 1.0e-12,
    "stress_spread_rel": 1.0e-11,
    "oracle_error_rel": 1.0e-11,
    "displacement_error": 1.0e-12,
    "deck_affine_check": 1.0e-12,
}


def run_case(formulation: str) -> dict[str, float]:
    topology, _ = CASES[formulation]
    directory = ROOT / formulation
    check_frozen_case(directory, formulation)
    prepared = prepare_analysis(directory / f"patch_{formulation}.toml")
    result = run_analysis(prepared)

    mesh = prepared.mesh
    expected_u = ((F_FINITE - np.eye(3)) @ mesh.X.T).T.ravel()
    final = assemble_internal(result.model, result.u, result.u, result.t, result.t, False)
    outputs = [g for block in final.gauss_output for g in block]
    interior = interior_dof_mask(mesh)
    force_scale = float(np.max(np.abs(final.f_int)))
    stresses = np.asarray([s for g in outputs for s in g.cauchy_stress])
    oracle = analytic_internal_force(mesh, topology, neo_stress(F_FINITE, NEO_KAPPA))
    return {
        "interior_residual_rel": float(np.max(np.abs(final.f_int[interior]))) / force_scale,
        "F_error": max(float(np.max(np.abs(F - F_FINITE))) for g in outputs for F in g.F_raw),
        "J_error": max(
            abs(J - float(np.linalg.det(F_FINITE))) for g in outputs for J in g.J_raw
        ),
        "stress_spread_rel": float(np.max(np.ptp(stresses, axis=0)))
        / float(np.max(np.abs(stresses))),
        "oracle_error_rel": float(np.max(np.abs(final.f_int - oracle)))
        / float(np.max(np.abs(oracle))),
        "displacement_error": float(np.max(np.abs(result.u - expected_u))),
        "deck_affine_check": float(result.verification["affine_deformation_error_inf"]),
        "newton_iterations_max": max(r.newton_iterations for r in result.increments),
        "cutbacks": sum(r.cutbacks for r in result.increments),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formulations", nargs="*", choices=[[], *CASES], default=list(CASES))
    args = parser.parse_args()
    formulations = args.formulations or list(CASES)

    failed = False
    rows = {name: run_case(name) for name in formulations}
    width = max(len(key) for key in THRESHOLDS) + 2
    header = "quantity".ljust(width) + "threshold".rjust(11)
    header += "".join(name.rjust(16) for name in formulations)
    print("\n" + header)
    print("-" * len(header))
    for key, threshold in THRESHOLDS.items():
        line = key.ljust(width) + f"{threshold:11.1e}"
        for name in formulations:
            value = rows[name][key]
            line += f"{value:16.2e}"
            failed = failed or value > threshold
        print(line)
    for key in ("newton_iterations_max", "cutbacks"):
        line = key.ljust(width) + " " * 11
        for name in formulations:
            line += f"{rows[name][key]:16d}"
        print(line)
    print(
        "\nResults kept under <case>/results/; for ParaView run\n"
        "  python -m fe_solver postprocess "
        "verification/patch_tests/<case>/results/run.h5"
    )
    if failed:
        print("\nPATCH TESTS FAILED: a quantity exceeded its threshold.")
        sys.exit(1)
    print("\nAll patch-test quantities within thresholds.")


if __name__ == "__main__":
    main()
