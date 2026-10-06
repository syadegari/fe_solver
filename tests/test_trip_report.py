from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from fe_solver.io import RESULT_SCHEMA_VERSION
from fe_solver.shape import HEX8_PARENT_NODES
from fe_solver.types import ModelError
from verification.check_periodic_composite import von_mises_stress
from verification.summarize_multiphase_trip import extract_trip_history, write_trip_report


class TripReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "run.h5"

    def _write_fixture(self, *, shear: bool = True) -> None:
        """Hand-built reporting snapshots, not a constitutive/equilibrium test."""
        unit_cube = (HEX8_PARENT_NODES + 1.0) / 2.0
        austenite = unit_cube.copy()
        austenite[:, 0] = 1.0 + 2.0 * austenite[:, 0]
        X = np.concatenate((unit_cube, austenite))
        x = X.copy()
        if shear:
            # Reference volumes 1 and 2; current volumes 1.2 and 1.8.
            x[:8, 0] = 1.2 * X[:8, 0] + 0.2 * X[:8, 1]
            x[8:, 0] = 1.2 + 0.9 * (X[8:, 0] - 1.0) + 0.2 * X[8:, 1]
            deformation = {
                "type": "simple_shear", "direction": "x", "normal": "y",
                "amount": {"curve": "ramp", "scale": 0.2},
            }
        else:
            x = X @ np.diag([1.2, 1 / np.sqrt(1.2), 1 / np.sqrt(1.2)])
            deformation = {
                "type": "isochoric_uniaxial", "axis": "x",
                "stretch": {"constant": 1.0, "curve": "ramp", "scale": 0.2},
            }
        resolved = {
            "analysis": {"t_start": 0.0, "t_end": 2000.0},
            "curves": [{"name": "ramp", "points": [[0.0, 0.0], [2000.0, 1.0]]}],
            "constraints": {"periodic_rve": [{"macro_deformation": deformation}]},
        }
        with h5py.File(self.database, "w") as h:
            h.attrs["schema_version"] = RESULT_SCHEMA_VERSION
            h.attrs["git_commit"] = "fixture"
            meta = h.create_group("meta")
            meta.attrs["analysis_name"] = "8a56f_fixture"
            meta.create_dataset("resolved_input_json", data=json.dumps(resolved))
            h.create_dataset("mesh/reference_coordinates", data=X)
            result = h.create_group("results")
            result.attrs["n_complete_steps"] = 2
            result.create_dataset("time", data=[0.0, 2000.0])
            result.create_dataset("nodal/displacement", data=np.stack((np.zeros_like(X), x - X)))
            result.create_dataset("nodal/constraint_reaction", data=np.zeros((2, 16, 3)))
            for index, region in enumerate(("ferrite", "austenite")):
                key = f"{index:04d}"
                block = h.create_group(f"mesh/blocks/{key}")
                block.attrs["region"] = region
                block.attrs["formulation"] = "hex8_fbar"
                block.create_dataset("connectivity", data=np.arange(8 * index, 8 * index + 8)[None, :])
                stress = np.zeros((2, 1, 6))
                stress[1, 0, 3 if shear else 0] = 0.1 * (index + 1)
                h.create_dataset(f"results/blocks/{key}/cauchy_stress", data=stress)
                state = h.create_group(f"results/blocks/{key}/state")
                state.create_dataset(f"beta_{region}", data=[[0.0], [0.05 * (index + 1)]])
                if region == "austenite":
                    state.create_dataset("martensite_fraction", data=[[0.0], [0.4]])

    def test_current_stress_reference_state_and_units(self) -> None:
        self._write_fixture()
        history = extract_trip_history(self.database)
        c = history["columns"]
        self.assertAlmostEqual(c["rve_cauchy_stress_12_MPa"][-1], 160.0)
        self.assertAlmostEqual(c["ferrite_cauchy_stress_12_MPa"][-1], 100.0)
        self.assertAlmostEqual(c["austenite_cauchy_stress_12_MPa"][-1], 200.0)
        self.assertAlmostEqual(history["initial_austenite_reference_volume_fraction"], 2 / 3)
        self.assertAlmostEqual(c["martensite_mean_initial_austenite_reference"][-1], 0.4)
        self.assertAlmostEqual(c["martensite_contribution_initial_rve_reference"][-1], 0.4 * 2 / 3)
        self.assertAlmostEqual(c["ferrite_beta_ferrite_mean_reference"][-1], 0.05)
        self.assertAlmostEqual(c["austenite_beta_austenite_mean_reference"][-1], 0.1)
        np.testing.assert_allclose(c["engineering_shear_gamma"], [0, 0.2])
        self.assertEqual(history["strain_column"], "twice_macro_hencky_12")
        self.assertAlmostEqual(c["macro_hencky_12"][-1], np.arcsinh(0.1) / np.sqrt(1.01))
        np.testing.assert_array_equal(c["twice_macro_hencky_12"], 2 * np.asarray(c["macro_hencky_12"]))
        self.assertAlmostEqual(c["macro_equivalent_hencky_strain"][-1], 2 * np.arcsinh(0.1) / np.sqrt(3))
        np.testing.assert_allclose(c["average_raw_F_12"], c["prescribed_macro_F_12"], atol=1e-14)
        self.assertLess(history["maximum_volume_average_F_error_inf"], 1e-14)
        self.assertEqual(history["beta_columns"]["ferrite"], "ferrite_beta_ferrite_mean_reference")
        self.assertEqual(history["display_name"], "8A56F_fixture")
        self.assertEqual(history["stress_components"], ["11", "22", "33", "12", "23", "13"])
        for label in history["stress_components"]:
            if label != "12":
                np.testing.assert_array_equal(c[f"rve_cauchy_stress_{label}_MPa"], [0.0, 0.0])
        self.assertAlmostEqual(c["rve_von_mises_of_mean_stress_MPa"][-1], np.sqrt(3) * 160)

    def test_von_mises_invariants(self) -> None:
        stress = np.array([
            np.diag([120.0, 0.0, 0.0]),
            np.eye(3) * 70.0,
            [[0.0, 50.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        np.testing.assert_allclose(von_mises_stress(stress), [120, 0, 50 * np.sqrt(3)])
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        np.testing.assert_allclose(von_mises_stress(rotation @ stress @ rotation.T), von_mises_stress(stress), atol=1e-13)

    def test_invariant_of_mean_is_not_mean_invariant(self) -> None:
        self._write_fixture()
        with h5py.File(self.database, "a") as h:
            h["results/blocks/0001/cauchy_stress"][1, 0, 3] = -0.2
        c = extract_trip_history(self.database)["columns"]
        self.assertAlmostEqual(c["rve_von_mises_of_mean_stress_MPa"][-1], np.sqrt(3) * 80)
        self.assertAlmostEqual(c["rve_mean_local_von_mises_stress_MPa"][-1], np.sqrt(3) * 160)

    def test_tension_uses_stretch_increment(self) -> None:
        self._write_fixture(shear=False)
        history = extract_trip_history(self.database)
        self.assertEqual(history["engineering_strain_column"], "engineering_axial_strain")
        np.testing.assert_allclose(history["columns"]["engineering_axial_strain"], [0, 0.2])
        np.testing.assert_allclose(history["columns"][history["strain_column"]], [0, np.log(1.2)])
        self.assertAlmostEqual(history["columns"]["macro_equivalent_hencky_strain"][-1], np.log(1.2))

    def test_uncommitted_tail_is_not_read(self) -> None:
        self._write_fixture()
        with h5py.File(self.database, "a") as h:
            h["results"].attrs["n_complete_steps"] = 1
            h["results/blocks/0001/state/martensite_fraction"][1] = np.nan
            h["results/blocks/0000/cauchy_stress"][1] = np.nan
        history = extract_trip_history(self.database)
        self.assertEqual(history["saved_end_time_s"], 0.0)
        self.assertEqual(history["requested_end_time_s"], 2000.0)
        self.assertEqual(history["columns"]["martensite_mean_initial_austenite_reference"], [0.0])

    def test_missing_transformation_output_is_not_zero_filled(self) -> None:
        self._write_fixture()
        with self.assertRaisesRegex(ModelError, "no scalar state output"):
            extract_trip_history(self.database, martensite_field="missing")

    def test_report_roundtrip(self) -> None:
        self._write_fixture()
        history = extract_trip_history(self.database)
        output = write_trip_report(history, Path(self.directory.name) / "plots")
        saved = json.loads((output / "history.json").read_text())
        self.assertEqual(saved, history)
        with np.load(output / "history.npz") as numeric:
            self.assertEqual(set(numeric.files), set(history["columns"]))
            for name, values in history["columns"].items():
                np.testing.assert_array_equal(numeric[name], values)
        for name in (
            "stress_vs_strain", "stress_components_vs_strain", "von_mises_vs_strain",
            "martensite_vs_strain", "beta_vs_strain",
        ):
            svg = output / f"{name}.svg"
            self.assertGreater(svg.stat().st_size, 1000)
            self.assertIn("2\\bar{h}_{12}", svg.read_text())
            self.assertNotIn("Prescribed macro", svg.read_text())
            self.assertNotIn("tensorial", svg.read_text())
            self.assertFalse((output / f"{name}.png").exists())


if __name__ == "__main__":
    unittest.main()
