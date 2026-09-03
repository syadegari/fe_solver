from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from fe_solver.assembly import build_model
from fe_solver.config import Deck, load_deck, mandatory_events
from fe_solver.constraints import build_constraints
from fe_solver.io import load_restart, write_restart
from fe_solver.mesh import read_gmsh
from fe_solver.solver import _factor_kkt, run_analysis
from fe_solver.types import RecoverableError


ROOT = Path(__file__).resolve().parents[1]


class MeshConstraintTests(unittest.TestCase):
    def test_gmsh_groups_and_periodic_spanning_forest(self) -> None:
        deck = load_deck(ROOT / "examples/case_b_hex8.toml")
        mesh = read_gmsh(deck.resolve(deck.data["mesh"]["file"]))
        self.assertEqual(len(mesh.X), 729)
        self.assertEqual(len(mesh.elements), 512)
        self.assertEqual(set(mesh.periodic_maps), {"xmax", "ymax", "zmax"})
        constraints = build_constraints(deck, mesh)
        self.assertEqual(constraints.C.shape, (654, 2187))
        affine_u = np.zeros_like(mesh.X)
        affine_u[:, 0] = 0.1 * mesh.X[:, 0]
        np.testing.assert_allclose(constraints.C @ affine_u.ravel(), constraints.rhs(1.0), atol=2e-15)

    def test_nonsymmetric_whole_kkt_solve(self) -> None:
        K = sparse.csr_matrix([[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.0, 2.0, 2.0]])
        C = sparse.csr_matrix([[1.0, 0.0, -1.0]])
        factor = _factor_kkt(K, C, "COLAMD")
        rhs = np.array([1.0, -2.0, 0.5, 0.3])
        solution = factor.solve(rhs)
        matrix = sparse.bmat([[K, C.T], [C, None]], format="csr")
        np.testing.assert_allclose(matrix @ solution, rhs, atol=2e-14)


class TimeRestartTests(unittest.TestCase):
    def test_mandatory_event_union(self) -> None:
        deck = load_deck(ROOT / "examples/case_a_hex8.toml")
        events, output, restart = mandatory_events(deck)
        np.testing.assert_allclose(events, np.arange(0.05, 1.0001, 0.05), atol=2e-15)
        self.assertIn(0.5, output)
        self.assertIn(0.5, restart)

    def test_restart_round_trip_and_identity(self) -> None:
        deck = load_deck(ROOT / "examples/case_a_hex8.toml")
        mesh = read_gmsh(deck.resolve(deck.data["mesh"]["file"]))
        model = build_model(deck, mesh)
        u = np.linspace(0.0, 0.01, mesh.ndof)
        lambdas = np.linspace(-1.0, 1.0, 36)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restart.npz"
            write_restart(path, model, 0.5, 0.075, u, lambdas)
            t, dt, loaded_u, loaded_lambdas = load_restart(path, model)
        self.assertEqual(t, 0.5)
        self.assertEqual(dt, 0.075)
        np.testing.assert_array_equal(loaded_u, u)
        np.testing.assert_array_equal(loaded_lambdas, lambdas)

    def test_restarted_and_uninterrupted_solutions_are_identical(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_data = copy.deepcopy(original.data)
            first_data["output"]["directory"] = str(root / "first")
            first_data["output"]["write_vtu"] = False
            first_data["restart"]["interval"] = 0.05
            first_data["restart"]["explicit_times"] = []
            first = Deck(original.path, first_data, original.curves)
            run_analysis(first, stop_time=0.05)
            restart_path = root / "first/restart/restart_000001.npz"

            resumed_data = copy.deepcopy(first_data)
            resumed_data["output"]["directory"] = str(root / "resumed")
            resumed_data["restart"]["restart_from"] = str(restart_path)
            resumed = run_analysis(Deck(original.path, resumed_data, original.curves), stop_time=0.1)

            cold_data = copy.deepcopy(first_data)
            cold_data["output"]["directory"] = str(root / "cold")
            cold_data["restart"]["restart_from"] = ""
            cold = run_analysis(Deck(original.path, cold_data, original.curves), stop_time=0.1)
        np.testing.assert_array_equal(resumed.u, cold.u)
        np.testing.assert_array_equal(resumed.lambdas, cold.lambdas)
        for resumed_block, cold_block in zip(resumed.model.blocks, cold.model.blocks):
            np.testing.assert_array_equal(resumed_block.state_n, cold_block.state_n)

    def test_recoverable_failure_cuts_back_from_committed_state(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        data = copy.deepcopy(original.data)
        data["output"]["directory"] = tempfile.mkdtemp(prefix="fe-cutback-")
        data["output"]["write_vtu"] = False
        data["restart"]["enabled"] = False
        deck = Deck(original.path, data, original.curves)
        from fe_solver import solver as solver_module
        actual_assemble = solver_module.assemble_internal
        calls = {"count": 0}

        def fail_once(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RecoverableError("deliberate trial failure")
            return actual_assemble(*args, **kwargs)

        with mock.patch("fe_solver.solver.assemble_internal", side_effect=fail_once):
            result = run_analysis(deck, stop_time=0.05)
        self.assertEqual(result.increments[0].cutbacks, 1)
        self.assertAlmostEqual(result.increments[0].t_np1, 0.025)
        self.assertAlmostEqual(result.t, 0.05)
        self.assertAlmostEqual(float(np.max(result.u)), 0.02, places=12)


if __name__ == "__main__":
    unittest.main()
