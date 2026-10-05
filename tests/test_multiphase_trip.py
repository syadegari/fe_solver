from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from fe_solver.multiphase_trip import BCC_STATE_LAYOUT, FCC_STATE_LAYOUT
from fe_solver.execution import ElementExecutor, ElementWorkItem
from fe_solver.materials import material_definition
from fe_solver.shape import HEX8_PARENT_NODES
from fe_solver.types import (
    FailureKind,
    MaterialInitRequest,
    MaterialRequest,
    ModelError,
    PointProperties,
)


_STUB_SOURCE = r"""
#include <math.h>

int cp_abi_version(void) { return 5; }
int cp_bcc_state_size(void) { return 93; }
int cp_fcc_state_size(void) { return 151; }
static int tangent_calls = 0;
int cp_tangent_calls(void) { return tangent_calls; }

static int initialize(int expected, int n, double *state) {
    if (n != expected) return 1;
    for (int i = 0; i < n; ++i) state[i] = 0.01 * (i + 1);
    return 0;
}
int cp_bcc_initialize(int n, double *state) { return initialize(93, n, state); }
int cp_fcc_initialize(int n, double *state) { return initialize(151, n, state); }

static double determinant(const double *F) {
    return F[0]*(F[4]*F[8] - F[5]*F[7])
         - F[1]*(F[3]*F[8] - F[5]*F[6])
         + F[2]*(F[3]*F[7] - F[4]*F[6]);
}

static int update(
    int expected, int n, const double *angles,
    double t_n, double t_np1, const double *F_n,
    const double *F_np1, const double *state_n, double *P, double *A,
    double *state_out, int need_tangent) {
    (void)F_n;
    if (n != expected) return 1;
    for (int i = 0; i < 3; ++i) if (!isfinite(angles[i])) return 2;
    if (!(t_np1 > t_n)) return 3;
    if (!(determinant(F_np1) > 0.0)) return 5;
    for (int i = 0; i < 9; ++i) P[i] = F_np1[i] - (i % 4 == 0 ? 1.0 : 0.0);
    if (need_tangent) ++tangent_calls;
    /* The wrapper must not expose or use an unrequested tangent buffer. */
    for (int i = 0; i < 81; ++i) A[i] = need_tangent ? 0.0 : NAN;
    if (need_tangent)
        for (int i = 0; i < 3; ++i)
            for (int I = 0; I < 3; ++I)
                A[27*i + 9*I + 3*i + I] = 1.0;
    for (int i = 0; i < n; ++i) state_out[i] = state_n[i] + 1.0;
    for (int i = 0; i < 3; ++i) state_out[i] += angles[i];
    return 0;
}

int cp_bcc_update(
    int n, const double *angles, double t0, double t1,
    const double *F0, const double *F1, const double *state0,
    double *P, double *A, double *state1, int need_tangent) {
    return update(93, n, angles, t0, t1,
                  F0, F1, state0, P, A, state1, need_tangent);
}
int cp_fcc_update(
    int n, const double *angles, double t0, double t1,
    const double *F0, const double *F1, const double *state0,
    double *P, double *A, double *state1, int need_tangent) {
    return update(151, n, angles, t0, t1,
                  F0, F1, state0, P, A, state1, need_tangent);
}
"""


class MultiphaseTRIPRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        source = cls.root / "stub.c"
        source.write_text(_STUB_SOURCE, encoding="utf-8")
        cls.library = cls.root / "libmultiphase_trip.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-O2", str(source), "-o", str(cls.library)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _definition(self, phase: str):
        return material_definition(
            {
                "name": phase,
                "model": (
                    "multiphase_trip_ferrite"
                    if phase == "bcc"
                    else "multiphase_trip_austenite"
                ),
                "properties": {
                    "library": self.library.name,
                },
            },
            base_directory=self.root,
        )

    def _initialize(self, phase: str, phase_id: int, orientation_id: int):
        definition = self._definition(phase)
        point_properties = PointProperties(
            (("phase_id", phase_id), ("orientation_id", orientation_id),
             ("euler_angles", np.array([0.1, 0.2, 0.3])))
        )
        response = definition.model.initialize(
            MaterialInitRequest(
                definition.properties, point_properties, np.zeros(3), 0.0
            )
        )
        return definition, point_properties, response

    def test_registered_layouts_and_relative_library_resolution(self) -> None:
        bcc = self._definition("bcc")
        fcc = self._definition("fcc")
        self.assertEqual(bcc.state_layout, BCC_STATE_LAYOUT)
        self.assertEqual(fcc.state_layout, FCC_STATE_LAYOUT)
        self.assertEqual(bcc.state_layout.n_state, 93)
        self.assertEqual(fcc.state_layout.n_state, 151)
        self.assertEqual(bcc.properties["library"], self.library.name)
        self.assertEqual(
            Path(str(bcc.properties["_resolved_library"])), self.library.resolve()
        )
        self.assertEqual(len(str(bcc.properties["library_sha256"])), 64)
        self.assertEqual(
            BCC_STATE_LAYOUT.fields[0].component_order[0:3], ("0", "1", "2")
        )
        self.assertEqual(FCC_STATE_LAYOUT.fields[0].component_order[-1], "150")

    def test_temperature_is_not_a_material_property(self) -> None:
        with self.assertRaisesRegex(ModelError, "unknown.*temperature"):
            material_definition(
                {
                    "name": "ferrite",
                    "model": "multiphase_trip_ferrite",
                    "properties": {"library": self.library.name, "temperature": 300.0},
                },
                base_directory=self.root,
            )

    def test_old_abi_is_rejected_before_initialization(self) -> None:
        source = self.root / "old_stub.c"
        source.write_text(
            _STUB_SOURCE.replace(
                "cp_abi_version(void) { return 5; }",
                "cp_abi_version(void) { return 4; }",
            ),
            encoding="utf-8",
        )
        library = self.root / "old_library.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-O2", str(source), "-o", str(library)],
            check=True,
        )
        definition = material_definition(
            {
                "name": "ferrite",
                "model": "multiphase_trip_ferrite",
                "properties": {"library": library.name},
            },
            base_directory=self.root,
        )
        initialized = definition.model.initialize(
            MaterialInitRequest(
                definition.properties,
                PointProperties((("phase_id", 1), ("euler_angles", np.zeros(3)))),
                np.zeros(3),
                0.0,
            )
        )
        self.assertEqual(initialized.status.kind, FailureKind.FATAL)
        self.assertIn("ABI version 4, expected 5; rebuild", initialized.status.message)

    def test_initialization_update_and_trial_state(self) -> None:
        for phase, phase_id, orientation_id, size in (
            ("bcc", 1, 1, 93),
            ("fcc", 2, 0, 151),
        ):
            definition, point_properties, initialized = self._initialize(
                phase, phase_id, orientation_id
            )
            self.assertTrue(initialized.status.ok)
            self.assertEqual(initialized.state0.shape, (size,))
            committed = initialized.state0.copy()
            F = np.array([[1.1, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            request = MaterialRequest(
                np.eye(3),
                F,
                definition.state_layout.view(initialized.state0),
                definition.properties,
                point_properties,
                0.0,
                0.1,
                True,
            )
            counter = ctypes.CDLL(str(self.library)).cp_tangent_calls
            before_tangent = counter()
            response = definition.model.update(request)
            self.assertEqual(counter(), before_tangent + 1)
            self.assertTrue(response.status.ok)
            np.testing.assert_array_equal(initialized.state0, committed)
            np.testing.assert_allclose(response.P, F - np.eye(3))
            expected_state = committed + 1.0
            expected_state[:3] += point_properties["euler_angles"]
            np.testing.assert_allclose(response.state_trial.values, expected_state)
            expected = np.einsum("ij,IJ->iIjJ", np.eye(3), np.eye(3))
            np.testing.assert_array_equal(response.A_alg, expected)

            without_tangent = definition.model.update(
                MaterialRequest(
                    request.F_n,
                    request.F_np1,
                    request.state_n,
                    request.properties,
                    request.point_properties,
                    request.t_n,
                    request.t_np1,
                    False,
                )
            )
            self.assertTrue(without_tangent.status.ok)
            self.assertEqual(counter(), before_tangent + 1)
            self.assertIsNone(without_tangent.A_alg)
            np.testing.assert_array_equal(without_tangent.P, response.P)
            np.testing.assert_array_equal(
                without_tangent.state_trial.values, response.state_trial.values
            )

    def test_phase_mismatch_is_fatal_and_bad_kinematics_recoverable(self) -> None:
        definition, _properties, initialized = self._initialize("bcc", 2, 0)
        self.assertEqual(initialized.status.kind, FailureKind.FATAL)

        definition, point_properties, initialized = self._initialize("bcc", 1, 0)
        self.assertTrue(initialized.status.ok)
        response = definition.model.update(
            MaterialRequest(
                np.eye(3),
                np.diag([-1.0, 1.0, 1.0]),
                definition.state_layout.view(initialized.state0),
                definition.properties,
                point_properties,
                0.0,
                0.1,
                True,
            )
        )
        self.assertEqual(response.status.kind, FailureKind.RECOVERABLE)

    def test_process_workers_lazy_load_library(self) -> None:
        definition, point_properties, initialized = self._initialize("bcc", 1, 0)
        self.assertTrue(initialized.status.ok)
        state_e = np.repeat(initialized.state0[None, :], 8, axis=0)
        X_e = 0.5 * (HEX8_PARENT_NODES + 1.0)
        items = [
            ElementWorkItem(
                0,
                index,
                X_e,
                np.zeros(24),
                np.zeros(24),
                state_e,
                point_properties,
                0.0,
                0.1,
                index == 0,
                "hex8",
            )
            for index in range(2)
        ]
        with ElementExecutor([definition], len(items), 2) as executor:
            responses = list(executor.evaluate(items))
        self.assertEqual(len(responses), 2)
        for index, response in enumerate(responses):
            self.assertTrue(response.status.ok)
            np.testing.assert_allclose(response.f_int, 0.0, atol=1.0e-14)
            if index == 0:
                self.assertIsNotNone(response.K)
            else:
                self.assertIsNone(response.K)
        np.testing.assert_array_equal(responses[0].state_trial, responses[1].state_trial)

    def test_angles_are_required_and_orientation_id_is_only_metadata(self) -> None:
        definition = self._definition("bcc")
        for properties in (
            {"phase_id": 1, "orientation_id": 0},
            {"phase_id": 1, "euler_angles": np.zeros(2)},
            {"phase_id": 1, "euler_angles": np.zeros((1, 3))},
            {"phase_id": 1, "euler_angles": [0., np.nan, 0.]},
        ):
            initialized = definition.model.initialize(
                MaterialInitRequest(definition.properties, properties, np.zeros(3), 0.)
            )
            self.assertEqual(initialized.status.kind, FailureKind.FATAL)
        # No catalog size or orientation ID is involved in the constitutive call.
        angles = np.array([0.12345678901234567, -0.4, 5.2], dtype=np.float64)
        properties = {"phase_id": 1, "euler_angles": angles}
        initialized = definition.model.initialize(
            MaterialInitRequest(definition.properties, properties, np.zeros(3), 0.)
        )
        self.assertTrue(initialized.status.ok)
        request = MaterialRequest(
            np.eye(3), np.eye(3), definition.state_layout.view(initialized.state0),
            definition.properties, properties, 0., .1, True,
        )
        response = definition.model.update(request)
        self.assertTrue(response.status.ok)
        np.testing.assert_array_equal(response.state_trial.values[:3], initialized.state0[:3] + 1. + angles)
        properties["orientation_id"] = 999999
        metadata_response = definition.model.update(request)
        np.testing.assert_array_equal(metadata_response.state_trial.values, response.state_trial.values)


if __name__ == "__main__":
    unittest.main()
