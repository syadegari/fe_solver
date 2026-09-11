from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from xml.etree import ElementTree as ET

import h5py
import numpy as np
from scipy import sparse

from fe_solver.assembly import assemble_internal, build_model, commit_trial_states
from fe_solver.config import Deck, load_deck, mandatory_events
from fe_solver.constraints import build_constraints, macro_deformation_function
from fe_solver.io import HDF5ResultWriter, load_restart, write_restart
from fe_solver.mesh import read_gmsh
from fe_solver.postprocess import write_xdmf
from fe_solver.output_fields import TENSOR_COMPONENTS, pack_symmetric, unpack_symmetric
from fe_solver.quadrature import HEX20_POINTS, HEX8_POINTS
from fe_solver.shape import hex20_shape, hex8_shape
from fe_solver.solver import _factor_kkt, run_analysis
from fe_solver.types import ModelError, RecoverableError
from verification.check_j2_prism import compare_prism_histories, extract_prism_history
from verification.run_j2_formulation_batch import (
    _parse_growth_thresholds,
    latest_restart,
)
from verification.run_j2_formulation_study import build_case_deck


ROOT = Path(__file__).resolve().parents[1]


class MeshConstraintTests(unittest.TestCase):
    def test_uniaxial_rotation_exact_between_events(self) -> None:
        deck = load_deck(ROOT / "examples/frame_objectivity_hex8.toml")
        entry = deck.data["constraints"]["affine"][0]
        self.assertEqual(entry["regions"], ["xmin", "xmax"])
        macro = macro_deformation_function(entry, deck)
        for t in (0.0, 0.023, 0.05, 0.1, 0.173, 0.55, 0.987, 1.0):
            F = macro(t)
            axial = 1.0 + min(t / 0.1, 1.0) * 0.1
            theta = np.deg2rad(max(0.0, (t - 0.1) / 0.9) * 90.0)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            U = R.T @ F
            np.testing.assert_allclose(U, np.diag(np.diag(U)), atol=1e-15)
            self.assertAlmostEqual(U[0, 0], axial)
            self.assertAlmostEqual(U[1, 1], U[2, 2])
            J = np.linalg.det(F)
            sigma = (F @ F.T - np.eye(3) + 20.0 * np.log(J) * np.eye(3)) / J
            axial_sigma = (axial**2 - U[1, 1]**2) / J
            np.testing.assert_allclose(sigma, axial_sigma * np.outer(R[:, 0], R[:, 0]), atol=2e-14)
        bad = copy.deepcopy(entry)
        bad["macro_deformation"]["material"] = "missing"
        with self.assertRaisesRegex(ModelError, "named neo_hook"):
            macro_deformation_function(bad, deck)

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

    def test_necking_mesh_groups_and_positive_reference_jacobians(self) -> None:
        mesh = read_gmsh(ROOT / "examples/necking_bar_quarter_hex8.msh")
        self.assertEqual(len(mesh.X), 1300)
        self.assertEqual(len(mesh.elements), 960)
        self.assertEqual(len(mesh.volume_groups["solid"]), 960)
        self.assertEqual(
            set(mesh.physical_dimensions),
            {
                "solid", "symmetry_x", "symmetry_y", "midplane",
                "loaded_end", "outer_surface", "neck_monitor",
            },
        )
        minimum_jacobian = np.inf
        for element in mesh.elements.values():
            X_e = mesh.X[element.connectivity]
            for point in HEX8_POINTS:
                _, dN = hex8_shape(point)
                minimum_jacobian = min(
                    minimum_jacobian, float(np.linalg.det(X_e.T @ dN))
                )
        self.assertGreater(minimum_jacobian, 0.0)

        prism = read_gmsh(ROOT / "examples/necking_prism_small_hex8.msh")
        self.assertEqual(len(prism.X), 225)
        self.assertEqual(len(prism.elements), 96)
        self.assertEqual(len(prism.volume_groups["solid"]), 96)
        self.assertEqual(set(prism.physical_dimensions), set(mesh.physical_dimensions))
        for element in prism.elements.values():
            X_e = prism.X[element.connectivity]
            for point in HEX8_POINTS:
                _, dN = hex8_shape(point)
                self.assertGreater(float(np.linalg.det(X_e.T @ dN)), 0.0)

        study_meshes = (
            ("necking_prism_refined_hex8.msh", 1225, 768, HEX8_POINTS, hex8_shape),
            ("necking_prism_small_hex20.msh", 741, 96, HEX20_POINTS, hex20_shape),
            (
                "necking_bar_quarter_refined_hex8.msh", 8967, 7680,
                HEX8_POINTS, hex8_shape,
            ),
        )
        for filename, expected_nodes, expected_elements, points, shape in study_meshes:
            with self.subTest(filename=filename):
                study = read_gmsh(ROOT / "examples" / filename)
                self.assertEqual(len(study.X), expected_nodes)
                self.assertEqual(len(study.elements), expected_elements)
                self.assertEqual(len(study.volume_groups["solid"]), expected_elements)
                self.assertEqual(
                    set(study.physical_dimensions), set(mesh.physical_dimensions)
                )
                for element in study.elements.values():
                    X_e = study.X[element.connectivity]
                    for point in points:
                        _, dN = shape(point)
                        self.assertGreater(float(np.linalg.det(X_e.T @ dN)), 0.0)

    def test_j2_formulation_study_case_matrix_is_orthogonal(self) -> None:
        def normalized(case_name: str) -> dict:
            data = copy.deepcopy(build_case_deck(case_name).data)
            data["analysis"]["name"] = "normalized"
            data["output"]["directory"] = "normalized"
            return data

        for fbar_name, standard_name in (
            ("coarse_hex8_fbar", "coarse_hex8"),
            ("refined_hex8_fbar", "refined_hex8"),
            ("soft_bulk_hex8_fbar", "soft_bulk_hex8"),
        ):
            with self.subTest(pair=(fbar_name, standard_name)):
                fbar = normalized(fbar_name)
                standard = normalized(standard_name)
                standard["element_assignments"][0]["formulation"] = "hex8_fbar"
                self.assertEqual(standard, fbar)

        coarse = normalized("coarse_hex8_fbar")
        refined = normalized("refined_hex8_fbar")
        refined["mesh"]["file"] = coarse["mesh"]["file"]
        self.assertEqual(refined, coarse)

        soft_bulk = normalized("soft_bulk_hex8_fbar")
        soft_bulk["materials"][0]["properties"]["bulk_modulus"] = (
            coarse["materials"][0]["properties"]["bulk_modulus"]
        )
        self.assertEqual(soft_bulk, coarse)

        hex20 = normalized("coarse_hex20")
        hex20["mesh"]["file"] = coarse["mesh"]["file"]
        hex20["element_assignments"][0]["formulation"] = "hex8_fbar"
        self.assertEqual(hex20, coarse)

        circular = normalized("circular_hex8_fbar")
        refined_circular = normalized("refined_circular_hex8_fbar")
        refined_circular["mesh"]["file"] = circular["mesh"]["file"]
        self.assertEqual(refined_circular, circular)

        deck = build_case_deck("coarse_hex20")
        mesh = read_gmsh(deck.resolve(deck.data["mesh"]["file"]))
        model = build_model(deck, mesh)
        self.assertEqual(model.blocks[0].state_n.shape[:2], (96, 27))

    def test_j2_formulation_study_isolates_output_and_selects_latest_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "isolated-output"
            deck = build_case_deck("coarse_hex20", output_directory=output)
            self.assertEqual(deck.data["output"]["directory"], str(output.resolve()))

            restart_directory = output / "restart"
            restart_directory.mkdir(parents=True)
            for index, restart_time in enumerate((0.1, 0.4, 0.5), start=1):
                with h5py.File(
                    restart_directory / f"restart_{index:06d}.h5", "w"
                ) as archive:
                    archive.attrs["t_n"] = restart_time
            self.assertEqual(
                latest_restart(output, at_most=0.45),
                restart_directory / "restart_000002.h5",
            )
            self.assertEqual(
                latest_restart(output, at_most=0.5),
                restart_directory / "restart_000003.h5",
            )

        baseline = build_case_deck("refined_hex8_fbar")
        adjusted = build_case_deck(
            "refined_hex8_fbar", grow_if_newton_iterations_le=6
        )
        self.assertEqual(
            baseline.data["time"]["grow_if_newton_iterations_le"], 5
        )
        self.assertEqual(
            adjusted.data["time"]["grow_if_newton_iterations_le"], 6
        )
        self.assertEqual(
            _parse_growth_thresholds(["refined_hex8_fbar=6"]),
            {"refined_hex8_fbar": 6},
        )
        with self.assertRaisesRegex(ValueError, "expected CASE=COUNT"):
            _parse_growth_thresholds(["refined_hex8_fbar"])

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
    @staticmethod
    def j2_data(original: Deck) -> dict:
        data = copy.deepcopy(original.data)
        data["materials"] = [{
            "name": "steel",
            "model": "j2_plasticity",
            "properties": {
                "shear_modulus": 80193.8,
                "bulk_modulus": 164210.0,
                "initial_yield_stress": 450.0,
                "linear_hardening_modulus": 129.24,
                "saturation_increment": 265.0,
                "saturation_rate": 16.93,
            },
        }]
        data["element_assignments"][0]["material"] = "steel"
        return data

    def test_j2_nonzero_state_restart_round_trip(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8_fbar.toml")
        deck = Deck(original.path, self.j2_data(original), original.curves)
        mesh = read_gmsh(deck.resolve(deck.data["mesh"]["file"]))
        model = build_model(deck, mesh)
        state = model.blocks[0].state_n
        state[..., :3] = [np.exp(-0.08), np.exp(0.04), np.exp(0.04)]
        state[..., 6] = 0.125
        expected = state.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "j2_restart.h5"
            write_restart(
                path, model, 0.4, 0.01, np.zeros(mesh.ndof),
                np.zeros(build_constraints(deck, mesh).C.shape[0]),
            )
            state[...] = 0.0
            load_restart(path, model)
        np.testing.assert_array_equal(state, expected)

    def test_j2_state_output_preserves_tensor_metadata(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8_fbar.toml")
        deck = Deck(original.path, self.j2_data(original), original.curves)
        mesh = read_gmsh(deck.resolve(deck.data["mesh"]["file"]))
        model = build_model(deck, mesh)
        u = np.zeros(mesh.ndof)
        assembly = assemble_internal(model, u, u, 0.0, 0.0, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "j2.h5"
            with HDF5ResultWriter(path, model) as writer:
                writer.append(0.0, u, u, assembly)
                state = writer.file["results/blocks/0000/state"]
                self.assertEqual(state["plastic_metric_inverse"].shape[-1], 6)
                self.assertEqual(
                    tuple(state["plastic_metric_inverse"].attrs["component_order"]),
                    TENSOR_COMPONENTS,
                )
                self.assertEqual(state["equivalent_plastic_strain"].shape[-1], len(model.blocks[0].connectivity))
            xdmf = write_xdmf(path)
            text = xdmf.read_text(encoding="utf-8")
            self.assertIn('Name="state_plastic_metric_inverse" AttributeType="Matrix"', text)
            self.assertIn('Name="component_order" Value="11,22,33,12,23,13"', text)

    def test_small_j2_prism_decks_are_paired_and_diagnostic_reads_database(self) -> None:
        fbar = load_deck(ROOT / "examples/j2_necking_prism_small_hex8_fbar.toml")
        standard = load_deck(ROOT / "examples/j2_necking_prism_small_hex8.toml")
        fbar_data = copy.deepcopy(fbar.data)
        standard_data = copy.deepcopy(standard.data)
        standard_data["analysis"]["name"] = fbar_data["analysis"]["name"]
        standard_data["element_assignments"][0]["formulation"] = "hex8_fbar"
        standard_data["output"]["directory"] = fbar_data["output"]["directory"]
        self.assertEqual(standard_data, fbar_data)

        mesh = read_gmsh(fbar.resolve(fbar.data["mesh"]["file"]))
        model = build_model(fbar, mesh)
        u_n = np.zeros(mesh.ndof)
        deformation = np.diag([0.99, 0.99, 1.02])
        u_trial = (mesh.X @ (deformation - np.eye(3)).T).ravel()
        initial = assemble_internal(model, u_n, u_n, 0.0, 0.0, False)
        trial = assemble_internal(model, u_n, u_trial, 0.0, 0.1, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prism.h5"
            with HDF5ResultWriter(path, model) as writer:
                writer.append(0.0, u_n, np.zeros_like(u_n), initial)
                commit_trial_states(model, trial.state_trial)
                writer.append(0.1, u_trial, np.zeros_like(u_n), trial)
            history = extract_prism_history(path)

        self.assertEqual(history.formulation, "hex8_fbar")
        self.assertEqual(len(history.time), 2)
        self.assertAlmostEqual(history.material_J_minimum[-1], np.linalg.det(deformation))
        self.assertAlmostEqual(history.material_J_maximum[-1], np.linalg.det(deformation))
        self.assertGreater(history.maximum_equivalent_plastic_strain[-1], 0.0)
        comparison = compare_prism_histories(history, history)
        for sample in comparison["samples"]:
            for field in sample["fields"].values():
                self.assertEqual(field["candidate_minus_reference"], 0.0)

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

    def test_in_place_resume_rolls_back_results_and_preserves_log_prefix(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        with tempfile.TemporaryDirectory() as directory:
            data = copy.deepcopy(original.data)
            data["output"]["directory"] = directory
            data["restart"].update(
                {"enabled": True, "interval": 0.05, "explicit_times": [], "keep_last": 3}
            )
            cold = run_analysis(Deck(original.path, data, original.curves), stop_time=0.1)
            output = Path(directory)
            database = output / "run.h5"
            log_path = output / "run_log.json"
            restart = output / "restart/restart_000001.h5"
            with log_path.open(encoding="utf-8") as stream:
                first_log = json.load(stream)
            with h5py.File(database, "r") as archive:
                np.testing.assert_array_equal(archive["results/time"], [0.0, 0.05, 0.1])

            resumed_data = copy.deepcopy(data)
            resumed_data["restart"]["restart_from"] = str(restart)
            resumed_data["time"]["grow_if_newton_iterations_le"] = (
                int(data["time"]["grow_if_newton_iterations_le"]) + 1
            )
            resumed = run_analysis(
                Deck(original.path, resumed_data, original.curves), stop_time=0.1
            )
            with log_path.open(encoding="utf-8") as stream:
                resumed_log = json.load(stream)
            with h5py.File(database, "r") as archive:
                np.testing.assert_array_equal(archive["results/time"], [0.0, 0.05, 0.1])
                self.assertEqual(int(archive["results"].attrs["n_complete_steps"]), 3)
                segments = []
                for value in archive["meta/input_segments_json"][:]:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    segments.append(json.loads(value))
                self.assertEqual([item["start_time"] for item in segments], [0.0, 0.05])
                self.assertEqual(
                    segments[0]["resolved_input"]["time"][
                        "grow_if_newton_iterations_le"
                    ],
                    data["time"]["grow_if_newton_iterations_le"],
                )
                self.assertEqual(
                    segments[1]["resolved_input"]["time"][
                        "grow_if_newton_iterations_le"
                    ],
                    resumed_data["time"]["grow_if_newton_iterations_le"],
                )

        self.assertEqual(len(resumed.increments), 2)
        self.assertEqual(len(resumed_log["increments"]), 2)
        self.assertEqual(resumed_log["increments"][0]["t_n"], 0.0)
        self.assertEqual(resumed_log["increments"][0]["t_np1"], 0.05)
        self.assertEqual(resumed_log["increments"][1]["t_np1"], 0.1)
        self.assertGreater(
            resumed_log["timing"]["elapsed_wall_seconds"],
            first_log["timing"]["elapsed_wall_seconds"],
        )
        np.testing.assert_array_equal(resumed.u, cold.u)
        np.testing.assert_array_equal(resumed.lambdas, cold.lambdas)

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

    def test_backtracking_rejects_invalid_full_newton_candidate(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        data = copy.deepcopy(original.data)
        with tempfile.TemporaryDirectory(prefix="fe-line-search-") as directory:
            data["output"]["directory"] = directory
            data["restart"]["enabled"] = False
            data["nonlinear"].update(
                {
                    "line_search": "backtracking",
                    "line_search_reduction": 0.5,
                    "line_search_armijo": 1.0e-4,
                    "line_search_min_alpha": 1.0e-4,
                    "line_search_max_backtracks": 14,
                }
            )
            deck = Deck(original.path, data, original.curves)
            from fe_solver import solver as solver_module
            actual_assemble = solver_module.assemble_internal
            rejected_full_candidate = False

            def reject_once(*args, **kwargs):
                nonlocal rejected_full_candidate
                # The first changed-displacement call is the alpha=1 candidate.
                # Mimic an inverted element so the search must try alpha=.5.
                need_tangent = args[5] if len(args) > 5 else kwargs["need_tangent"]
                if (
                    not rejected_full_candidate
                    and args[4] > args[3]
                    and need_tangent
                    and not np.array_equal(args[1], args[2])
                ):
                    rejected_full_candidate = True
                    raise RecoverableError("deliberate invalid full-step candidate")
                return actual_assemble(*args, **kwargs)

            with mock.patch("fe_solver.solver.assemble_internal", side_effect=reject_once):
                result = run_analysis(deck, stop_time=0.05)
        self.assertTrue(rejected_full_candidate)
        self.assertEqual(result.increments[0].cutbacks, 0)
        first_iteration = next(
            row for row in result.newton_history
            if row["t_n"] == 0.0 and row["t_np1"] == 0.05 and row["iteration"] == 0
        )
        self.assertEqual(first_iteration["line_search_alpha"], 0.5)
        self.assertEqual(first_iteration["line_search_backtracks"], 1)
        self.assertAlmostEqual(result.t, 0.05)

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
                    archive["results/blocks/0000/green_lagrange_strain"][-1, 0], pack_symmetric(green)
                )
                np.testing.assert_allclose(
                    archive["results/blocks/0000/euler_almansi_strain"][-1, 0], pack_symmetric(almansi)
                )
                assembly = assemble_internal(result.model, result.u, result.u, result.t, result.t, False)
                expected_stress = np.mean(assembly.gauss_output[0][0].cauchy_stress, axis=0)
                np.testing.assert_allclose(
                    archive["results/blocks/0000/cauchy_stress"][-1, 0], pack_symmetric(expected_stress)
                )
                dataset = archive["results/nodal/displacement"]
                dataset.resize(count + 1, axis=0)
            model = build_model(deck, read_gmsh(deck.resolve(deck.data["mesh"]["file"])))
            with HDF5ResultWriter(path, model, resume=True) as writer:
                self.assertEqual(writer.file["results/nodal/displacement"].shape[0], count)
            xdmf = write_xdmf(path, Path(directory) / "view/case.xdmf")
            text = xdmf.read_text(encoding="utf-8")
            self.assertIn("case.xdmf.h5:/steps/000001/results/nodal/displacement", text)
            self.assertNotIn("HyperSlab", text)
            with h5py.File(xdmf.with_suffix(".xdmf.h5"), "r") as visual:
                ds = visual["steps/000001/results/nodal/displacement"]
                self.assertTrue(ds.is_virtual)
                self.assertEqual(ds.id.get_storage_size(), 0)
                np.testing.assert_array_equal(ds[:].ravel(), result.u)
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
                np.testing.assert_allclose(block["cauchy_stress"][-1, 0], pack_symmetric(expected))

    def test_tensorial_six_component_output_and_xdmf_views(self) -> None:
        # All shears nonzero and distinct: detect 13/23 swaps and engineering scaling.
        tensor = np.array([[1., 4., 6.], [4., 2., 5.], [6., 5., 3.]])
        np.testing.assert_array_equal(pack_symmetric(tensor), [1., 2., 3., 4., 5., 6.])
        np.testing.assert_array_equal(unpack_symmetric(pack_symmetric(tensor)), tensor)
        deck = load_deck(ROOT / "examples/case_a_hex8.toml")
        mesh = read_gmsh(deck.resolve(deck.data["mesh"]["file"]))
        model = build_model(deck, mesh)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.h5"
            with HDF5ResultWriter(path, model) as writer:
                for t in (0., 0.4, 1.):
                    F = np.eye(3) + t * np.array([[.1, .2, .07], [0., -.02, .13], [0., 0., .03]])
                    u = (mesh.X @ (F - np.eye(3)).T).ravel()
                    assembly = assemble_internal(model, np.zeros_like(u), u, 0., t, False)
                    writer.append(t, u, np.zeros_like(u), assembly)
                    inverse = np.linalg.inv(F)
                    expected = {
                        "green_lagrange_strain": .5 * (F.T @ F - np.eye(3)),
                        "euler_almansi_strain": .5 * (np.eye(3) - inverse.T @ inverse),
                        "cauchy_stress": np.mean(assembly.gauss_output[0][0].cauchy_stress, axis=0),
                    }
                    for name, matrix in expected.items():
                        ds = writer.file[f"results/blocks/0000/{name}"]
                        self.assertEqual(ds.shape[1:], (32, 6))
                        self.assertEqual(tuple(ds.attrs["component_order"]), TENSOR_COMPONENTS)
                        self.assertEqual(ds.attrs["shear_scale"], 1.)
                        np.testing.assert_allclose(ds[-1, 0], pack_symmetric(matrix), atol=1e-14)
            xdmf = write_xdmf(path)
            grids = ET.parse(xdmf).findall("./Domain/Grid/Grid")
            with h5py.File(path, "r") as archive, h5py.File(xdmf.with_suffix(".xdmf.h5"), "r") as visual:
                for step, grid in enumerate(grids):
                    for name in expected:
                        full = grid.find(f"./Grid/Attribute[@Name='{name}']")
                        self.assertEqual(full.attrib["AttributeType"], "Matrix")
                        self.assertEqual(full.find("DataItem").attrib["Dimensions"], "32 6")
                        item = full.find("DataItem")
                        self.assertEqual(item.attrib["Format"], "HDF")
                        self.assertEqual(item.attrib["NumberType"], "Float")
                        ds = visual[item.text.split(":", 1)[1]]
                        self.assertTrue(ds.is_virtual)
                        self.assertEqual(ds.id.get_storage_size(), 0)
                        np.testing.assert_array_equal(ds[:], archive[f"results/blocks/0000/{name}"][step])
                        for label in TENSOR_COMPONENTS:
                            self.assertIsNone(grid.find(f"./Grid/Attribute[@Name='{name}_{label}']"))


if __name__ == "__main__":
    unittest.main()
