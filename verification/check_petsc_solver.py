"""Bounded MPI checks, including rank-zero element workers and both Newton modes.

Run with an MPI/PETSc-enabled environment:
  mpiexec --bind-to none -n 2 python -m verification.check_petsc_solver
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse

from fe_solver.config import Deck, load_deck
from fe_solver.linear_solver import LinearSolverConfig, factor_kkt
from fe_solver.petsc_solver import PetscSession
from fe_solver.solver import _run_analysis, run_analysis
from .compare_run_databases import compare_databases


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="artifact directory (default: a temporary directory)")
    parser.add_argument("--num-processes", type=int, default=2, help="rank-zero element workers")
    args = parser.parse_args()
    # Keep MPI imports inside main: multiprocessing children must not initialize
    # MPI when Python reconstructs this entry module via spawn/forkserver.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    directory = None
    if comm.rank == 0:
        directory = args.output or Path(tempfile.mkdtemp(prefix="fe-petsc-check-"))
        directory.mkdir(parents=True, exist_ok=True)
    directory = Path(comm.bcast(str(directory) if comm.rank == 0 else None, root=0))
    report = {"mpi_ranks": comm.size, "checks": []}

    try:
        # Compare overlap 0 and 1, nonsymmetric KKT solves, context reuse and
        # zero RHS. This verifies mechanics, not large-mesh overlap robustness.
        for overlap in (0, 1):
            options = directory / f"overlap_{overlap}.toml"
            if comm.rank == 0:
                options.write_text(f"fieldsplit_u_pc_asm_overlap = {overlap}\n")
            with PetscSession(options) as session:
                if comm.rank != 0:
                    status = session.serve()
                    if status:
                        raise SystemExit(status)
                    continue
                K = sparse.diags([np.full(11, -.2), np.full(12, 4.), np.full(11, .7)], [-1, 0, 1], format="csr")
                C = sparse.csr_matrix(([1., -1., 1., -1.], ([0, 0, 1, 1], [0, 3, 6, 10])), shape=(2, 12))
                config = LinearSolverConfig("petsc_lgmres", petsc_session=session)
                context = factor_kkt(K, C, "COLAMD", config)
                reference = factor_kkt(K, C, "COLAMD")
                try:
                    for rhs in (np.linspace(-1., 1., 14), np.linspace(2., -1., 14), np.zeros(14)):
                        np.testing.assert_allclose(context.solve(rhs), reference.solve(rhs), rtol=1e-8, atol=1e-10)
                        report["checks"].append({"overlap": overlap, **context.diagnostics})
                finally:
                    context.close()

        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        for method in ("newton", "modified_newton"):
            data = copy.deepcopy(original.data)
            data["nonlinear"]["method"] = method
            data["nonlinear"]["line_search"] = "backtracking"
            data["restart"]["enabled"] = False
            baseline_dir, petsc_dir = directory / f"{method}_scipy", directory / f"{method}_petsc"
            if comm.rank == 0:
                data["output"]["directory"] = str(baseline_dir)
                # Deliberately root-only direct reference within this MPI test;
                # ordinary multi-rank direct CLI launches are rejected.
                baseline = _run_analysis(
                    Deck(original.path, data, original.curves), stop_time=.05,
                    num_processes=args.num_processes, debug_timing=False,
                    linear_solver=LinearSolverConfig("scipy_splu"),
                )
            data["output"]["directory"] = str(petsc_dir)
            candidate = run_analysis(
                Deck(original.path, data, original.curves), stop_time=.05,
                num_processes=args.num_processes, solver_backend="petsc_lgmres",
            )
            if comm.rank == 0:
                comparison = compare_databases(baseline_dir / "run.h5", petsc_dir / "run.h5", rtol=1e-8, atol=1e-8)
                if not comparison.passed:
                    raise AssertionError(comparison.as_dict())
                np.testing.assert_allclose(candidate.lambdas, baseline.lambdas, rtol=1e-8, atol=1e-8)
                report["checks"].append({"method": method, "database_comparison": comparison.as_dict()})

        if comm.rank == 0:
            (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"PETSc bounded checks passed; results: {directory}")
    except BaseException:
        # A failed assertion on root must not leave peers awaiting another test.
        if comm.size > 1:
            comm.Abort(2)
        raise


if __name__ == "__main__":
    main()
