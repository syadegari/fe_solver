"""Run the normative neo-Hookean acceptance decks and report numerical checks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from fe_solver.assembly import assemble_internal
from fe_solver.config import Deck, load_deck
from fe_solver.solver import AnalysisResult, run_analysis


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECKS = [
    "case_a_hex8.toml", "case_a_hex8_fbar.toml", "case_a_hex20.toml",
    "case_b_hex8.toml", "case_b_hex8_fbar.toml",
]


def run_deck(name: str, output_root: Path) -> AnalysisResult:
    source = load_deck(ROOT / "examples" / name)
    data = copy.deepcopy(source.data)
    data["output"]["directory"] = str(output_root / Path(name).stem)
    return run_analysis(Deck(source.path, data, source.curves))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", action="append", choices=DEFAULT_DECKS)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/fe_solver_acceptance"))
    args = parser.parse_args()
    names = args.deck or DEFAULT_DECKS
    results: dict[str, AnalysisResult] = {}
    summary: dict[str, dict] = {}
    for name in names:
        result = run_deck(name, args.output_root)
        results[name] = result
        summary[name] = {
            "increments": len(result.increments),
            "maximum_newton_iterations": max(item.newton_iterations for item in result.increments),
            "total_cutbacks": sum(item.cutbacks for item in result.increments),
            **result.verification,
        }
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
