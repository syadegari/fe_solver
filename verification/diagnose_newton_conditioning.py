"""Replay one small-model Newton attempt and inspect its reduced tangent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve

from fe_solver.assembly import assemble_internal, commit_trial_states
from fe_solver.io import load_restart
from fe_solver.preprocess import prepare_analysis
from fe_solver.solver import _newton_attempt
from fe_solver.types import RecoverableError


def _constraint_nullspace(C: sparse.csr_matrix) -> np.ndarray:
    return linalg.null_space(C.toarray())


def _recover_multipliers(C: sparse.csr_matrix, reaction: np.ndarray) -> np.ndarray:
    return np.asarray(spsolve(C @ C.T, -C @ reaction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path)
    parser.add_argument("restart", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--committed-time", type=float)
    parser.add_argument("--attempt-time", type=float, required=True)
    parser.add_argument(
        "--line-search", choices=("deck", "none", "backtracking"), default="deck",
        help="optionally override the deck's nonlinear.line_search setting",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepared = prepare_analysis(args.deck)
    if args.line_search != "deck":
        prepared.deck.data["nonlinear"]["line_search"] = args.line_search
    t_n, _, u_n, lambda_n = load_restart(args.restart, prepared.model)
    with h5py.File(args.database, "r") as archive:
        complete = int(archive["results"].attrs["n_complete_steps"])
        times = np.asarray(archive["results/time"][:complete])
        displacements = archive["results/nodal/displacement"]
        reactions = archive["results/nodal/constraint_reaction"]
        replay_limit = (
            args.committed_time
            if args.committed_time is not None
            else float(times[times < args.attempt_time - 1.0e-12].max())
        )
        replay_times = times[
            (times > t_n + 1.0e-12) & (times <= replay_limit + 1.0e-12)
        ]
        for time in replay_times:
            index = int(np.flatnonzero(np.isclose(times, time, rtol=0.0, atol=1.0e-12))[0])
            u_trial = np.asarray(displacements[index]).ravel()
            assembly = assemble_internal(
                prepared.model, u_n, u_trial, t_n, float(time), False
            )
            commit_trial_states(prepared.model, assembly.state_trial)
            t_n, u_n = float(time), u_trial
            lambda_n = _recover_multipliers(
                prepared.constraints.C, np.asarray(reactions[index]).ravel()
            )
    committed_state_source = "database"
    committed_setup_iterations: int | None = None
    if args.committed_time is not None and t_n < args.committed_time - 1.0e-12:
        setup_history: list[dict] = []
        setup = _newton_attempt(
            prepared.model, prepared.constraints, u_n, lambda_n,
            t_n, args.committed_time, setup_history,
        )
        commit_trial_states(prepared.model, setup.assembly.state_trial)
        t_n, u_n, lambda_n = args.committed_time, setup.u, setup.lambdas
        committed_state_source = "solved_from_preceding_database_state"
        committed_setup_iterations = setup.iterations
    if args.committed_time is not None and abs(t_n - args.committed_time) > 1.0e-12:
        raise RuntimeError(f"cannot reconstruct committed state at t={args.committed_time}")

    Z = _constraint_nullspace(prepared.constraints.C)
    diagnostics = []

    def observer(record: dict, assembly) -> None:
        if assembly.K is None:
            return
        reduced = Z.T @ assembly.K @ Z
        singular = np.linalg.svd(reduced, compute_uv=False)
        diagnostics.append(
            {
                "iteration": record["iteration"],
                "force_residual_inf": record["force_residual_inf"],
                "constraint_residual_inf": record["constraint_residual_inf"],
                "largest_singular_value": float(singular[0]),
                "smallest_singular_value": float(singular[-1]),
                "condition_number": float(singular[0] / singular[-1]),
            }
        )

    history = []
    outcome = "converged"
    failure = ""
    try:
        result = _newton_attempt(
            prepared.model, prepared.constraints, u_n, lambda_n,
            t_n, args.attempt_time, history, observer,
        )
        final_time = args.attempt_time
        iterations = result.iterations
    except RecoverableError as exc:
        outcome = "recoverable_failure"
        failure = str(exc)
        final_time = t_n
        iterations = None
    report = {
        "committed_time": t_n,
        "committed_state_source": committed_state_source,
        "committed_setup_iterations": committed_setup_iterations,
        "attempt_time": args.attempt_time,
        "line_search": str(prepared.deck.data["nonlinear"].get("line_search", "none")),
        "outcome": outcome,
        "failure": failure,
        "iterations": iterations,
        "reduced_dimension": int(Z.shape[1]),
        "conditioning": diagnostics,
        "newton_history": history,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
