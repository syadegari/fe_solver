from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest import mock

import numpy as np
import h5py
from scipy import sparse
from scipy.sparse.linalg import splu

from fe_solver.config import Deck, load_deck
from fe_solver.linear_solver import LinearSolverConfig, factor_kkt
from fe_solver.petsc_solver import PetscSession, load_petsc_options, partition
from fe_solver.solver import run_analysis
from fe_solver.types import ModelError, RecoverableError


ROOT = Path(__file__).resolve().parents[1]
HAS_PETSC = all(importlib.util.find_spec(name) is not None for name in ("petsc4py", "mpi4py"))


class PetscIntegrationTests(unittest.TestCase):
    def test_flat_options_override_defaults(self):
        defaults = load_petsc_options()
        self.assertEqual(defaults["fieldsplit_u_pc_asm_overlap"], 1)
        self.assertEqual(defaults["fieldsplit_u_sub_pc_factor_levels"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "options.toml"
            path.write_text("fieldsplit_u_pc_asm_overlap = 0\n")
            options = load_petsc_options(path)
            self.assertEqual(options["fieldsplit_u_pc_asm_overlap"], 0)
            self.assertEqual(options["ksp_rtol"], 1e-9)
            for invalid in ("[nested]\nx = 1", "ksp_rtol = nan", "x = [1, 2]"):
                path.write_text(invalid)
                with self.assertRaises(ModelError):
                    load_petsc_options(path)

    def test_partitions_are_complete_even_with_empty_ranks(self):
        for n in (2, 19):
            for size in (1, 2, 4):
                pieces = [partition(n, size, rank) for rank in range(size)]
                self.assertEqual([i for lo, hi in pieces for i in range(lo, hi)], list(range(n)))
                counts = [hi - lo for lo, hi in pieces]
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_csr_dispatch_preserves_nonsymmetric_kkt(self):
        K = sparse.csr_matrix([[3., 1.], [-2., 4.]])
        C = sparse.csr_matrix([[1., -1.]])
        session = mock.Mock()
        factor = factor_kkt(K, C, "COLAMD", LinearSolverConfig("petsc_lgmres", petsc_session=session))
        self.assertIs(factor, session.setup.return_value)
        matrix, n = session.setup.call_args.args
        self.assertEqual(matrix.format, "csr")
        self.assertEqual(n, 2)
        np.testing.assert_array_equal(matrix.toarray(), [[3., 1., 1.], [-2., 4., -1.], [1., -1., 0.]])
        with self.assertRaisesRegex(ModelError, "active PetscSession"):
            factor_kkt(K, C, "COLAMD", LinearSolverConfig("petsc_lgmres"))

    def test_nonroot_does_not_preprocess_or_write(self):
        session = mock.MagicMock()
        session.__enter__.return_value = session
        session.rank = 1
        session.serve.return_value = 0
        with (
            mock.patch("fe_solver.petsc_solver.PetscSession", return_value=session),
            mock.patch("fe_solver.solver.prepare_analysis", side_effect=AssertionError("nonroot preprocessing")),
        ):
            result = run_analysis(ROOT / "examples/case_a_hex8.toml", solver_backend="petsc_lgmres")
        self.assertIsNone(result)
        session.serve.assert_called_once()
        session.serve.return_value = 2
        with mock.patch("fe_solver.petsc_solver.PetscSession", return_value=session):
            with self.assertRaises(SystemExit) as failure:
                run_analysis(ROOT / "examples/case_a_hex8.toml", solver_backend="petsc_lgmres")
        self.assertEqual(failure.exception.code, 2)

    def test_newton_context_reuse_and_logging_without_petsc(self):
        # SciPy stand-in verifies routing/lifecycle, not PETSc numerical accuracy.
        class Session:
            rank = 0
            info = {"mpi_ranks": 2, "threads_per_rank": 1}

            def __init__(self, path):
                self.factor = None
                self.setups = self.closes = 0
                self.diagnostics = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                if self.factor is not None:
                    raise AssertionError("leaked factor")

            def setup(self, matrix, ndof):
                if self.factor is not None:
                    raise AssertionError("replaced live factor")
                self.factor = splu(matrix.tocsc())
                self.setups += 1
                return self

            def solve(self, rhs):
                self.diagnostics = {"iterations": 1, "converged_reason": 2}
                return self.factor.solve(rhs)

            def close(self):
                if self.factor is not None:
                    self.closes += 1
                    self.factor = None

        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        with tempfile.TemporaryDirectory() as directory:
            for method in ("newton", "modified_newton"):
                data = copy.deepcopy(original.data)
                data["nonlinear"]["method"] = method
                data["output"]["directory"] = str(Path(directory) / method)
                data["restart"]["enabled"] = False
                session = Session(None)
                with mock.patch("fe_solver.petsc_solver.PetscSession", return_value=session):
                    result = run_analysis(Deck(original.path, data, original.curves), stop_time=.05, solver_backend="petsc_lgmres")
                self.assertEqual(session.closes, session.setups)
                self.assertGreater(session.setups, 0)
                solves = [row["linear_solve"] for row in result.newton_history if "linear_solve" in row]
                self.assertTrue(solves)
                self.assertEqual(result.analysis["linear_system"]["options"]["mpi_ranks"], 2)
                if method == "modified_newton":
                    self.assertEqual(session.setups, sum(row.attempts for row in result.increments))

    def test_failed_accuracy_is_recoverable_and_block_residuals_are_recorded(self):
        session = object.__new__(PetscSession)
        session.n, session.displacement_dofs = 3, 2
        session.comm = mock.Mock()
        session._root_matrix = sparse.eye(3, format="csr")
        session._collective = mock.Mock(return_value=np.array([1., 2., 0.]))
        session.diagnostics = {
            "converged_reason": 2, "true_residual_l2": .1,
            "true_relative_residual": .01, "residual_tolerance": 1e-9,
        }
        with self.assertRaises(RecoverableError):
            session.solve(np.array([1., 2., .1]))
        self.assertEqual(session.diagnostics["force_linear_residual_inf"], 0.)
        self.assertEqual(session.diagnostics["constraint_linear_residual_inf"], .1)

    def test_failed_solves_are_saved_and_terminal_error_preserves_cause(self):
        # Inject a KSP failure through the real Newton/cutback path; no PETSc
        # installation is needed to verify persistence and cleanup.
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        message = "PETSc KKT solve failed accuracy/convergence check: reason=-3, true relative residual=2.000e-08"
        diagnostics = {
            "iterations": 5000, "converged_reason": -3,
            "reported_residual_l2": 2e-8, "true_residual_l2": 2e-8,
            "true_relative_residual": 2e-8, "residual_tolerance": 1e-9,
            "force_linear_residual_inf": 1e-8,
            "constraint_linear_residual_inf": 1e-10,
        }
        for method in ("newton", "modified_newton"):
            for limit in ("dt_min", "maximum increment attempts"):
                with self.subTest(method=method, limit=limit), tempfile.TemporaryDirectory() as directory:
                    data = copy.deepcopy(original.data)
                    data["nonlinear"]["method"] = method
                    data["time"].update({
                        "dt_initial": .05, "dt_max": .05,
                        "dt_min": .025 if limit == "dt_min" else 1e-5,
                        "max_attempts_per_increment": 2,
                    })
                    data["output"]["directory"] = directory
                    data["restart"]["enabled"] = False
                    session = mock.MagicMock()
                    session.__enter__.return_value = session
                    session.rank = 0
                    session.info = {"mpi_ranks": 4, "threads_per_rank": 1}
                    session.setup.return_value = session
                    session.diagnostics = diagnostics
                    session.solve.side_effect = RecoverableError(message)
                    with mock.patch("fe_solver.petsc_solver.PetscSession", return_value=session):
                        with self.assertRaisesRegex(ModelError, limit) as failure:
                            run_analysis(
                                Deck(original.path, data, original.curves),
                                stop_time=.05, solver_backend="petsc_lgmres",
                            )
                    self.assertIn(message, str(failure.exception))
                    self.assertIsInstance(failure.exception.__cause__, RecoverableError)
                    self.assertEqual(session.close.call_count, 2)
                    with (Path(directory) / "run_log.json").open() as stream:
                        log = json.load(stream)
                    self.assertEqual(log["increments"], [])
                    self.assertGreater(log["timing"]["elapsed_wall_seconds"], 0.)
                    failures = log["newton_history"]
                    self.assertEqual(len(failures), 2)
                    for attempt, row in enumerate(failures, 1):
                        self.assertEqual(row["increment_attempt"], attempt)
                        self.assertEqual(row["failure_kind"], "recoverable")
                        self.assertEqual(row["failure_message"], message)
                        self.assertEqual(row["linear_solve"], diagnostics)
                    with h5py.File(Path(directory) / "run.h5", "r") as archive:
                        np.testing.assert_array_equal(archive["results/time"], [0.])

    def test_help_works_without_optional_dependencies(self):
        for command in (
            [sys.executable, "native/benchmark_petsc_kkt.py", "--help"],
            [sys.executable, "-m", "fe_solver", "solve", "--help"],
        ):
            output = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout
            self.assertIn("usage:", output)

    def test_benchmark_selects_database_and_group_and_forwards_petsc_flags(self):
        petsc_module, mpi_module = ModuleType("petsc4py"), ModuleType("mpi4py")
        petsc_module.init, petsc_module.PETSc = mock.Mock(), mock.Mock()
        mpi_module.MPI = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arbitrary_name.h5"
            K, C = sparse.eye(2, format="csr"), sparse.csr_matrix([[1., -1.]])
            with h5py.File(path, "w") as h5:
                for name, scale in (("unused", 10.), ("chosen/group", 2.)):
                    group = h5.create_group(name)
                    for field, matrix in (("K", K * scale), ("C", C)):
                        item = group.create_group(field)
                        item.attrs["shape"] = matrix.shape
                        for key in ("data", "indices", "indptr"):
                            item.create_dataset(key, data=getattr(matrix, key))
                    group.create_dataset("r_u", data=[scale, -scale])
                    group.create_dataset("r_c", data=[.5])
            with (
                mock.patch.dict(sys.modules, {"petsc4py": petsc_module, "mpi4py": mpi_module}),
                mock.patch.dict(os.environ),
                mock.patch.object(sys, "argv", ["benchmark", str(path), "--group", "chosen/group", "-bench_ksp_rtol", "1e-8"]),
            ):
                module = runpy.run_path(str(ROOT / "native/benchmark_petsc_kkt.py"))
            petsc_module.init.assert_called_once_with([
                str(ROOT / "native/benchmark_petsc_kkt.py"), "-bench_ksp_rtol", "1e-8",
            ])
            args = module["_args"]
            k, c, matrix, rhs = module["load_problem"](args.database, args.group)
            np.testing.assert_array_equal(k.toarray(), 2. * np.eye(2))
            np.testing.assert_array_equal(rhs, [-2., 2., -.5])
            self.assertEqual(matrix.shape, (3, 3))

    def test_direct_backend_never_initializes_petsc(self):
        with (
            mock.patch("fe_solver.petsc_solver.PetscSession", side_effect=AssertionError("eager PETSc")),
            mock.patch("fe_solver.solver._run_analysis", return_value="direct") as solve,
        ):
            self.assertEqual(run_analysis(ROOT / "examples/case_a_hex8.toml"), "direct")
        solve.assert_called_once()
        with self.assertRaisesRegex(ModelError, "only with petsc_lgmres"):
            run_analysis(ROOT / "examples/case_a_hex8.toml", petsc_options="unused.toml")

    @unittest.skipUnless(HAS_PETSC, "PETSc/mpi4py are not installed")
    def test_real_petsc_repeated_rhs_and_zero_rhs(self):
        K = sparse.csr_matrix([[3., 2., 0.], [-1., 4., 1.], [0., 2., 2.]])
        C = sparse.csr_matrix([[1., 0., -1.]])
        session = PetscSession()
        if session.size != 1:
            self.skipTest("use verification.check_petsc_solver for multi-rank checks")
        with session:
            context = factor_kkt(K, C, "COLAMD", LinearSolverConfig("petsc_lgmres", petsc_session=session))
            reference = factor_kkt(K, C, "COLAMD")
            try:
                for rhs in (np.array([1., -2., .5, .3]), np.array([-2., 1., 4., 0.]), np.zeros(4)):
                    np.testing.assert_allclose(context.solve(rhs), reference.solve(rhs), rtol=1e-8, atol=1e-10)
            finally:
                context.close()


if __name__ == "__main__":
    unittest.main()
