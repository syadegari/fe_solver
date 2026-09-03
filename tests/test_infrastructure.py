from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
from scipy import sparse

from fe_solver.assembly import assemble_internal, build_model
from fe_solver.config import Deck, load_deck, mandatory_events
from fe_solver.constraints import build_constraints, macro_deformation_function
from fe_solver.io import HDF5ResultWriter, load_restart, write_restart
from fe_solver.mesh import read_gmsh
from fe_solver.postprocess import write_xdmf
from fe_solver.shape import hex8_shape
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

    def test_heterogeneous_periodic_mesh_partition(self) -> None:
        mesh = read_gmsh(ROOT / "examples/heterogeneous_periodic_cube_8x8x8.msh")
        self.assertEqual(len(mesh.X), 729)
        self.assertEqual(len(mesh.elements), 512)
        self.assertEqual(len(mesh.volume_groups["core"]), 64)
        self.assertEqual(len(mesh.volume_groups["matrix"]), 448)
        self.assertEqual(set(mesh.periodic_maps), {"xmax", "ymax", "zmax"})

    def test_affine_boundary_and_exact_macro_paths(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        mesh = read_gmsh(original.resolve(original.data["mesh"]["file"]))
        data = copy.deepcopy(original.data)
        data["constraints"] = {
            "affine": [{
                "regions": ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"],
                "origin": [0.0, 0.0, 0.0],
                "macro_deformation": {
                    "type": "isochoric_uniaxial", "axis": "x",
                    "stretch": {"constant": 1.0, "curve": "ramp", "scale": 0.2},
                },
            }]
        }
        deck = Deck(original.path, data, original.curves)
        macro = macro_deformation_function(data["constraints"]["affine"][0], deck)
        np.testing.assert_allclose(macro(1.0), np.diag([1.2, 1 / np.sqrt(1.2), 1 / np.sqrt(1.2)]))
        constraints = build_constraints(deck, mesh)
        boundary = np.unique(np.concatenate([mesh.node_groups[name] for name in data["constraints"]["affine"][0]["regions"]]))
        self.assertEqual(constraints.C.shape[0], 3 * len(boundary))
        u = mesh.X @ (macro(1.0) - np.eye(3)).T
        np.testing.assert_allclose(constraints.C @ u.ravel(), constraints.rhs(1.0), atol=2e-15)

        shear_entry = {
            "macro_deformation": {
                "type": "simple_shear", "direction": "x", "normal": "y",
                "amount": {"curve": "ramp", "scale": 0.2},
            }
        }
        expected = np.eye(3)
        expected[0, 1] = 0.2
        np.testing.assert_allclose(macro_deformation_function(shear_entry, deck)(1.0), expected)

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
            path = Path(directory) / "restart.h5"
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
            first_data["restart"]["interval"] = 0.05
            first_data["restart"]["explicit_times"] = []
            first = Deck(original.path, first_data, original.curves)
            run_analysis(first, stop_time=0.05)
            restart_path = root / "first/restart/restart_000001.h5"

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
        with tempfile.TemporaryDirectory(prefix="fe-cutback-") as directory:
            data["output"]["directory"] = directory
            data["restart"]["enabled"] = False
            deck = Deck(original.path, data, original.curves)
            from fe_solver import solver as solver_module
            actual_assemble = solver_module.assemble_internal
            calls = {"count": 0}

            def fail_once(*args, **kwargs):
                calls["count"] += 1
                # Call one recovers the initial accepted state for output.
                if calls["count"] == 2:
                    raise RecoverableError("deliberate trial failure")
                return actual_assemble(*args, **kwargs)

            with mock.patch("fe_solver.solver.assemble_internal", side_effect=fail_once):
                result = run_analysis(deck, stop_time=0.05)
        self.assertEqual(result.increments[0].cutbacks, 1)
        self.assertAlmostEqual(result.increments[0].t_np1, 0.025)
        self.assertAlmostEqual(result.t, 0.05)
        self.assertAlmostEqual(float(np.max(result.u)), 0.02, places=12)

    def test_hdf5_result_schema_centroid_recovery_and_tail_truncation(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(original.data)
            data["output"]["directory"] = directory
            data["restart"]["enabled"] = False
            deck = Deck(original.path, data, original.curves)
            result = run_analysis(deck, stop_time=0.05)
            path = Path(directory) / "run.h5"
            with h5py.File(path, "r+") as archive:
                count = int(archive["results"].attrs["n_complete_steps"])
                self.assertEqual(count, len(result.increments) + 1)
                np.testing.assert_allclose(archive["results/time"], [0.0, 0.05])
                self.assertNotIn("F", archive["results/blocks/0000"])
                self.assertNotIn("current_coordinates", archive["mesh"])
                displacement = archive["results/nodal/displacement"][-1]
                np.testing.assert_allclose(displacement.ravel(), result.u)
                connectivity = result.model.blocks[0].connectivity[0]
                X = result.model.mesh.X[connectivity]
                x = X + displacement[connectivity]
                _, dN = hex8_shape(np.zeros(3))
                F = (x.T @ dN) @ np.linalg.inv(X.T @ dN)
                green = 0.5 * (F.T @ F - np.eye(3))
                Finv = np.linalg.inv(F)
                almansi = 0.5 * (np.eye(3) - Finv.T @ Finv)
                np.testing.assert_allclose(
                    archive["results/blocks/0000/green_lagrange_strain"][-1, 0], green
                )
                np.testing.assert_allclose(
                    archive["results/blocks/0000/euler_almansi_strain"][-1, 0], almansi
                )
                assembly = assemble_internal(result.model, result.u, result.u, result.t, result.t, False)
                expected_stress = np.mean(assembly.gauss_output[0][0].cauchy_stress, axis=0)
                np.testing.assert_allclose(
                    archive["results/blocks/0000/cauchy_stress"][-1, 0], expected_stress
                )
                dataset = archive["results/nodal/displacement"]
                dataset.resize(count + 1, axis=0)
            model = build_model(deck, read_gmsh(deck.resolve(deck.data["mesh"]["file"])))
            with HDF5ResultWriter(path, model, resume=True) as writer:
                self.assertEqual(writer.file["results/nodal/displacement"].shape[0], count)
            xdmf = write_xdmf(path, Path(directory) / "view/case.xdmf")
            text = xdmf.read_text(encoding="utf-8")
            self.assertIn("../run.h5:/results/nodal/displacement", text)
            self.assertEqual(text.count("<Time "), count)
            self.assertTrue(xdmf.with_suffix(".xdmf.h5").is_file())

    def test_fbar_database_recovers_stress_without_storing_F(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8_fbar.toml")
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(original.data)
            data["output"]["directory"] = directory
            data["restart"]["enabled"] = False
            result = run_analysis(Deck(original.path, data, original.curves), stop_time=0.05)
            assembly = assemble_internal(result.model, result.u, result.u, result.t, result.t, False)
            expected = np.mean(assembly.gauss_output[0][0].cauchy_stress, axis=0)
            with h5py.File(Path(directory) / "run.h5", "r") as archive:
                block = archive["results/blocks/0000"]
                self.assertNotIn("F", block)
                self.assertEqual(block["cauchy_stress"].attrs["recovery"], "gauss_interpolation")
                np.testing.assert_allclose(block["cauchy_stress"][-1, 0], expected)


if __name__ == "__main__":
    unittest.main()
