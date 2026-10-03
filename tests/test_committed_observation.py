"""Startup/restart reporting must not invoke a constitutive evolution map."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from fe_solver.assembly import assemble_internal, build_model, commit_trial_states, observe_committed
from fe_solver.config import Deck, load_deck
from fe_solver.io import load_restart, write_restart
from fe_solver.material_point import run_material_path
from fe_solver.materials import evaluate_material_point, neo_hook_definition
from fe_solver.mesh import read_gmsh
from fe_solver.output_fields import pack_symmetric
from fe_solver.solver import run_analysis
from fe_solver.types import ModelError


ROOT = Path(__file__).resolve().parents[1]


class CommittedObservationTests(unittest.TestCase):
    def test_observation_matches_accepted_response_for_all_elements(self) -> None:
        rng = np.random.default_rng(17)
        for formulation in ("hex8", "hex8_fbar", "hex20"):
            with self.subTest(formulation=formulation):
                deck = load_deck(ROOT / f"examples/case_a_{formulation}.toml")
                model = build_model(deck, read_gmsh(deck.resolve(deck.data["mesh"]["file"])))
                u = 0.015 * rng.normal(size=model.mesh.ndof)
                accepted = assemble_internal(model, np.zeros_like(u), u, 0., 0.1, False)
                commit_trial_states(model, accepted.state_trial, accepted.gauss_output)
                original_state = [block.state_n.copy() for block in model.blocks]
                with mock.patch("fe_solver.elements.evaluate_material_point", side_effect=AssertionError("update")):
                    observed = observe_committed(model, u)
                self.assertIsNone(observed.K)
                np.testing.assert_allclose(observed.f_int, accepted.f_int, rtol=2e-14, atol=2e-14)
                for accepted_block, observed_block in zip(accepted.gauss_output, observed.gauss_output):
                    for a, b in zip(accepted_block, observed_block):
                        for name in ("F_raw", "F_material", "J_raw", "J_material", "P_material", "P_effective", "cauchy_stress"):
                            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=2e-13, atol=2e-13)
                for block, state in zip(model.blocks, original_state):
                    np.testing.assert_array_equal(block.state_n, state)

    def test_cold_start_and_restart_only_update_positive_intervals(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8_fbar.toml")
        for method in ("newton", "modified_newton"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = copy.deepcopy(original.data)
                mesh_path = root / "one_element.msh"
                subprocess.run(
                    [sys.executable, str(ROOT / "examples/generate_elongated_block_gmsh.py"),
                     "--nx", "1", "--ny", "1", "--nz", "1", "--out", str(mesh_path)],
                    check=True, capture_output=True, timeout=30,
                )
                data["mesh"]["file"] = str(mesh_path)
                data["time"].update(dt_min=0.01, max_attempts_per_increment=4)
                data["nonlinear"]["method"] = method
                data["output"]["directory"] = str(root / "first")
                data["restart"].update(interval=0.05, explicit_times=[], keep_last=3)
                data["materials"] = [{"name": "matrix", "model": "j2_plasticity", "properties": {
                    "shear_modulus": 80193.8, "bulk_modulus": 164210., "initial_yield_stress": 450.,
                    "linear_hardening_modulus": 129.24, "saturation_increment": 265., "saturation_rate": 16.93,
                }}]
                # The absolute material scale requires the usual benchmark force tolerance.
                data["nonlinear"].update(force_atol=1e-6, max_iterations=30, line_search="backtracking")
                calls = []

                def strictly_positive(material, request):
                    self.assertGreater(request.t_np1, request.t_n)
                    calls.append((request.t_n, request.t_np1, request.need_tangent))
                    return evaluate_material_point(material, request)

                with mock.patch("fe_solver.elements.evaluate_material_point", side_effect=strictly_positive):
                    first = run_analysis(Deck(original.path, data, original.curves), stop_time=0.05)
                    restart = root / "first/restart/restart_000001.h5"
                    observation_data = copy.deepcopy(data)
                    observation_data["restart"]["restart_from"] = str(restart)
                    observation_data["output"]["directory"] = str(root / "observation_only")
                    observation_data["verification"].update(
                        check_element_directional_tangent=False, check_global_directional_tangent=False,
                    )
                    with mock.patch("fe_solver.elements.evaluate_material_point", side_effect=AssertionError("update")):
                        observed = run_analysis(Deck(original.path, observation_data, original.curves), stop_time=0.05)
                    np.testing.assert_array_equal(observed.u, first.u)
                    resumed_data = copy.deepcopy(data)
                    resumed_data["output"]["directory"] = str(root / "resumed")
                    resumed_data["restart"]["restart_from"] = str(restart)
                    resumed = run_analysis(Deck(original.path, resumed_data, original.curves), stop_time=0.1)
                    cold_data = copy.deepcopy(data)
                    cold_data["output"]["directory"] = str(root / "cold")
                    cold = run_analysis(Deck(original.path, cold_data, original.curves), stop_time=0.1)
                self.assertTrue(any(need for _, _, need in calls))
                self.assertGreater(first.model.blocks[0].state_n[..., 6].max(), 0.)
                np.testing.assert_allclose(resumed.u, cold.u, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(resumed.model.blocks[0].state_n, cold.model.blocks[0].state_n, rtol=1e-12, atol=1e-12)
                with h5py.File(root / "first/run.h5", "r") as a, h5py.File(root / "resumed/run.h5", "r") as b:
                    np.testing.assert_array_equal(a["results/blocks/0000/cauchy_stress"][-1], b["results/blocks/0000/cauchy_stress"][0])
                    np.testing.assert_array_equal(a["results/blocks/0000/cauchy_stress"][0], 0.)
                with h5py.File(root / "observation_only/run.h5", "r") as archive:
                    self.assertEqual(len(archive["results/time"]), 1)
                    np.testing.assert_array_equal(
                        archive["results/blocks/0000/cauchy_stress"][0],
                        pack_symmetric(np.mean(first.model.blocks[0].cauchy_stress_n, axis=1)),
                    )
                with h5py.File(restart, "r") as archive:
                    self.assertEqual(archive.attrs["schema_version"], 3)
                    self.assertEqual(archive["cauchy_stress/0000"].shape, (1, 8, 6))
                    self.assertNotIn("tangent", archive)

    def test_restart_stress_validation_and_no_partial_restore(self) -> None:
        deck = load_deck(ROOT / "examples/case_a_hex8.toml")
        model = build_model(deck, read_gmsh(deck.resolve(deck.data["mesh"]["file"])))
        stress = model.blocks[0].cauchy_stress_n
        stress[..., 0, 0] = 5.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restart.h5"
            write_restart(path, model, 0.5, 0.05, np.zeros(model.mesh.ndof), np.empty(0))
            stress[...] = 0.
            load_restart(path, model)
            self.assertEqual(stress[..., 0, 0].min(), 5.)
            with h5py.File(path, "a") as archive:
                archive["cauchy_stress/0000"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ModelError, "stress layout/values"):
                load_restart(path, model)
            self.assertEqual(stress[..., 0, 0].min(), 5.)
            with h5py.File(path, "a") as archive:
                archive.attrs["schema_version"] = 2
            with self.assertRaisesRegex(ModelError, "version 3"):
                load_restart(path, model)

    def test_material_point_initial_row_is_observation(self) -> None:
        material = neo_hook_definition("elastic", {"mu": 2., "kappa": 12.})
        with mock.patch("fe_solver.material_point.evaluate_material_point", wraps=evaluate_material_point) as update:
            history = run_material_path(material, np.diag([1.1, 1., 1.]), 3)
        self.assertEqual(update.call_count, 3)
        for call in update.call_args_list:
            request = call.args[1]
            self.assertGreater(request.t_np1, request.t_n)
        np.testing.assert_array_equal(history.P[0], 0.)
        self.assertTrue(np.isnan(history.A_alg[0]).all())
        self.assertTrue(np.isfinite(history.A_alg[1:]).all())
        with self.assertRaisesRegex(ModelError, "reference F=I"):
            run_material_path(material, lambda s: np.diag([1.1 + s, 1., 1.]), 1)
