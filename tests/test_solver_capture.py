from __future__ import annotations

import copy
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np
from scipy import sparse

from fe_solver.config import Deck, load_deck
from fe_solver.solver import run_analysis
from fe_solver import solver_capture
from fe_solver.types import ModelError
from native import superlu_mt_backend


ROOT = Path(__file__).resolve().parents[1]
with mock.patch.dict(sys.modules, {"superlu_mt_backend": superlu_mt_backend}):
    load_superlu_problem = runpy.run_path(
        str(ROOT / "native/benchmark_superlu_mt.py")
    )["load_problem"]


class SolverCaptureTests(unittest.TestCase):
    def test_existing_benchmark_readers_load_capture_without_changes(self):
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        K = sparse.csr_matrix([[3., 1.], [-2., 4.]])
        C = sparse.csr_matrix([[1., -1.]])
        r_u, r_c = np.array([2., -3.]), np.array([.01])
        expected = sparse.bmat([[K, C.T], [C, None]], format="csr")
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(original.data)
            data["output"]["directory"] = directory
            model = SimpleNamespace(deck=Deck(original.path, data, original.curves))
            path = solver_capture._capture_kkt_system(model, K, C, r_u, r_c, 0., .1, 0)
            group_name = "t_n=0.0/iteration=0"
            with h5py.File(path, "r") as archive:
                group = archive[group_name]
                self.assertEqual(set(group), {"K", "C", "r_u", "r_c"})
                self.assertEqual(group.attrs["t_np1"], .1)
                for name, matrix in (("K", K), ("C", C)):
                    item = group[name]
                    self.assertEqual(set(item), {"data", "indices", "indptr"})
                    np.testing.assert_array_equal(item.attrs["shape"], matrix.shape)
                    self.assertEqual(item["data"].dtype, np.dtype("float64"))
                    self.assertEqual(item["indices"].dtype, np.dtype("int32"))
                    self.assertEqual(item["indptr"].dtype, np.dtype("int32"))
            matrix, rhs = load_superlu_problem(path, group_name)
            np.testing.assert_array_equal(matrix.toarray(), expected.toarray())
            np.testing.assert_array_equal(rhs, -np.concatenate((r_u, r_c)))

            # Exercise the actual PETSc benchmark's HDF5 reader without loading
            # MPI/PETSc: this checks file compatibility, not the iterative solve.
            petsc, mpi = ModuleType("petsc4py"), ModuleType("mpi4py")
            petsc.init, petsc.PETSc, mpi.MPI = mock.Mock(), mock.Mock(), mock.Mock()
            with (
                mock.patch.dict(sys.modules, {"petsc4py": petsc, "mpi4py": mpi}),
                mock.patch.dict(os.environ, {"BENCH_MODE": "validate"}),
                mock.patch.object(sys, "argv", ["benchmark", str(path), "--group", group_name]),
            ):
                benchmark = runpy.run_path(str(ROOT / "native/benchmark_petsc_kkt.py"))
            k, c, matrix, rhs = benchmark["load_problem"](path, group_name)
            np.testing.assert_array_equal(k.toarray(), K.toarray())
            np.testing.assert_array_equal(c.toarray(), C.toarray())
            np.testing.assert_array_equal(matrix.toarray(), expected.toarray())
            np.testing.assert_array_equal(rhs, -np.concatenate((r_u, r_c)))
            with self.assertRaisesRegex(ModelError, "capture already exists"):
                solver_capture._capture_kkt_system(model, K, C, r_u, r_c, 0., .05, 0)
            # Refusing a second capture must preserve the original endpoint.
            with h5py.File(path, "r") as archive:
                self.assertEqual(archive[group_name].attrs["t_np1"], .1)

    def test_first_two_systems_and_corrections_match_actual_newton_operators(self):
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        for method in ("newton", "modified_newton"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                data = copy.deepcopy(original.data)
                data["nonlinear"]["method"] = method
                data["nonlinear"]["line_search"] = "backtracking"
                data["restart"]["enabled"] = False
                data["output"]["directory"] = str(Path(directory) / "reference")
                reference = run_analysis(Deck(original.path, data, original.curves), stop_time=.1)
                data["output"]["directory"] = str(Path(directory) / "capture")
                with mock.patch.object(
                    solver_capture, "_capture_kkt_system", wraps=solver_capture._capture_kkt_system,
                ) as capture:
                    result = solver_capture.run_analysis(
                        Deck(original.path, data, original.curves), stop_time=.1,
                    )
                np.testing.assert_array_equal(result.u, reference.u)
                np.testing.assert_array_equal(result.lambdas, reference.lambdas)
                self.assertEqual(
                    [vars(row) for row in result.increments],
                    [vars(row) for row in reference.increments],
                )
                self.assertEqual(capture.call_count, 2)
                folder = Path(data["output"]["directory"]) / "kkt_capture"
                self.assertEqual(len(list(folder.glob("*.h5"))), 2)
                for iteration, call in enumerate(capture.call_args_list):
                    _, K, C, r_u, r_c, t_n, t_np1, k = call.args
                    self.assertEqual((t_n, k), (0., iteration))
                    self.assertEqual(t_np1, result.increments[0].t_np1)
                    path = folder / f"kkt_debug_t0_iter{iteration}.h5"
                    group_name = f"t_n=0.0/iteration={iteration}"
                    matrix, rhs = load_superlu_problem(path, group_name)
                    expected = sparse.bmat([[K, C.T], [C, None]], format="csc")
                    np.testing.assert_array_equal(matrix.toarray(), expected.toarray())
                    np.testing.assert_array_equal(rhs, -np.concatenate((r_u, r_c)))
                    with h5py.File(path, "r") as archive:
                        correction = np.asarray(archive[group_name]["correction"])
                    self.assertLess(np.linalg.norm(matrix @ correction - rhs, ord=np.inf), 1e-10)
                if method == "modified_newton":
                    np.testing.assert_array_equal(
                        capture.call_args_list[0].args[1].data,
                        capture.call_args_list[1].args[1].data,
                    )


if __name__ == "__main__":
    unittest.main()
