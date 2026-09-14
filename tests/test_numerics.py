from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from fe_solver.elements import evaluate_element, evaluate_fbar_reference_element
from fe_solver.j2_kernel import available_j2_backends, configure_j2_backend, numba_signatures
from fe_solver.materials import (
    evaluate_material_point,
    init_j2_plasticity,
    j2_plasticity_definition,
    material_definition,
    neo_hook_definition,
    register_material_model,
    update_j2_plasticity,
    update_neo_hook,
)
from fe_solver.material_point import run_material_path
from fe_solver.shape import HEX20_PARENT_NODES, HEX8_PARENT_NODES, hex20_shape, hex8_shape
from fe_solver.types import (
    ElementRequest,
    EvaluationStatus,
    MaterialDefinition,
    MaterialInitRequest,
    MaterialInitResponse,
    MaterialModel,
    MaterialRequest,
    MaterialResponse,
    ModelError,
    StateField,
    StateLayout,
)


_PROBE_LAYOUT = StateLayout((StateField("accumulated", (2,)),))


def init_history_probe(request: MaterialInitRequest) -> MaterialInitResponse:
    return MaterialInitResponse(np.array([request.X[0], request.t0]), EvaluationStatus())


def update_history_probe(request: MaterialRequest) -> MaterialResponse:
    values = request.state_n.values + float(request.properties["increment"])
    return MaterialResponse(
        np.zeros((3, 3)), np.zeros((3, 3, 3, 3)) if request.need_tangent else None,
        _PROBE_LAYOUT.view(values), EvaluationStatus(),
    )


class ShapeTests(unittest.TestCase):
    def test_partition_derivatives_and_nodes(self) -> None:
        rng = np.random.default_rng(12)
        for shape, nodes in ((hex8_shape, HEX8_PARENT_NODES), (hex20_shape, HEX20_PARENT_NODES)):
            for xi in rng.uniform(-1, 1, size=(20, 3)):
                N, dN = shape(xi)
                self.assertAlmostEqual(float(N.sum()), 1.0, places=13)
                np.testing.assert_allclose(dN.sum(axis=0), 0.0, atol=2e-14)
                np.testing.assert_allclose(N @ nodes, xi, atol=2e-14)
            for a, xi in enumerate(nodes):
                N, _ = shape(xi)
                expected = np.zeros(len(nodes))
                expected[a] = 1.0
                np.testing.assert_allclose(N, expected, atol=2e-14)


class MaterialTests(unittest.TestCase):
    @staticmethod
    def j2_properties() -> dict[str, float]:
        return {
            "shear_modulus": 80193.8,
            "bulk_modulus": 164210.0,
            "initial_yield_stress": 450.0,
            "linear_hardening_modulus": 129.24,
            "saturation_increment": 265.0,
            "saturation_rate": 16.93,
        }

    def j2_initial_state(self):
        model = j2_plasticity_definition("steel", self.j2_properties())
        initialized = init_j2_plasticity(
            MaterialInitRequest(model.properties, None, np.zeros(3), 0.0)
        )
        return model, model.state_layout.view(initialized.state0)

    def test_model_properties_and_history_are_independent(self) -> None:
        soft = material_definition(
            {"name": "soft", "model": "neo_hook", "properties": {"mu": 1.0, "kappa": 20.0}}
        )
        stiff = material_definition(
            {"name": "stiff", "model": "neo_hook", "properties": {"mu": 10.0, "kappa": 200.0}}
        )
        self.assertIs(soft.model, stiff.model)
        self.assertIsNone(soft.model.initialize)
        self.assertEqual(soft.state_layout.n_state, 0)
        self.assertNotEqual(soft.properties, stiff.properties)

    def test_registered_stateful_model_uses_named_state(self) -> None:
        root = "history_probe"
        try:
            register_material_model(
                MaterialModel(
                    root, update_history_probe, init_history_probe,
                    lambda values: {"increment": float(values["increment"])}, _PROBE_LAYOUT,
                )
            )
        except Exception as exc:
            if "already registered" not in str(exc):
                raise
        model = material_definition(
            {"name": "probe", "model": root, "properties": {"increment": 0.25}}
        )
        initialized = model.model.initialize(
            MaterialInitRequest(model.properties, None, np.array([2.0, 0.0, 0.0]), 0.5)
        )
        state = model.state_layout.view(initialized.state0)
        response = evaluate_material_point(
            model,
            MaterialRequest(np.eye(3), np.eye(3), state, model.properties, None, 0.5, 1.0, False),
        )
        np.testing.assert_allclose(response.state_trial["accumulated"], [2.25, 0.75])

    def test_neo_hookean_tangent(self) -> None:
        rng = np.random.default_rng(7)
        F = np.eye(3) + 0.15 * rng.normal(size=(3, 3))
        params = {"mu": 2.3, "kappa": 17.0}
        empty_state = StateLayout().view(np.empty(0))
        req = MaterialRequest(np.eye(3), F, empty_state, params, None, 0.0, 1.0, True)
        result = update_neo_hook(req)
        self.assertTrue(result.status.ok)
        dF = rng.normal(size=(3, 3))
        dF /= np.linalg.norm(dF)
        eps = 2e-7
        pm = update_neo_hook(
            MaterialRequest(np.eye(3), F - eps * dF, empty_state, params, None, 0, 1, False)
        ).P
        pp = update_neo_hook(
            MaterialRequest(np.eye(3), F + eps * dF, empty_state, params, None, 0, 1, False)
        ).P
        analytic = np.einsum("iIjJ,jJ->iI", result.A_alg, dF)
        np.testing.assert_allclose(analytic, (pp - pm) / (2 * eps), rtol=2e-8, atol=2e-8)
        no_tangent = update_neo_hook(
            MaterialRequest(np.eye(3), F, empty_state, params, None, 0, 1, False)
        )
        np.testing.assert_array_equal(result.P, no_tangent.P)
        self.assertIsNone(no_tangent.A_alg)

    def test_neo_hookean_superposed_rotation(self) -> None:
        angle = 0.73
        Q = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        F = np.array([[1.12, 0.08, 0.0], [0.0, 0.96, 0.03], [0.0, 0.0, 1.01]])
        properties = {"mu": 2.3, "kappa": 17.0}
        state = StateLayout().view(np.empty(0))
        response = update_neo_hook(
            MaterialRequest(np.eye(3), F, state, properties, None, 0.0, 1.0, False)
        )
        rotated = update_neo_hook(
            MaterialRequest(np.eye(3), Q @ F, state, properties, None, 0.0, 1.0, False)
        )
        np.testing.assert_allclose(rotated.P, Q @ response.P, atol=2e-14)
        sigma = response.P @ F.T / np.linalg.det(F)
        sigma_rotated = rotated.P @ (Q @ F).T / np.linalg.det(Q @ F)
        np.testing.assert_allclose(sigma_rotated, Q @ sigma @ Q.T, atol=2e-14)

    def test_periodic_composite_materials_match_reference_elasticity(self) -> None:
        mu = 80193.8
        bulk_modulus = 164210.0
        neo_properties = {
            "mu": mu,
            "kappa": bulk_modulus - 2.0 * mu / 3.0,
        }
        empty_state = StateLayout().view(np.empty(0))
        neo = update_neo_hook(
            MaterialRequest(
                np.eye(3), np.eye(3), empty_state, neo_properties,
                None, 0.0, 0.0, True,
            )
        )
        j2_model, j2_state = self.j2_initial_state()
        j2 = update_j2_plasticity(
            MaterialRequest(
                np.eye(3), np.eye(3), j2_state, j2_model.properties,
                None, 0.0, 0.0, True,
            )
        )
        np.testing.assert_allclose(neo.A_alg, j2.A_alg, rtol=0.0, atol=2.0e-11)

    def test_j2_initialization_hydrostatic_response_and_properties(self) -> None:
        model, state = self.j2_initial_state()
        self.assertEqual(model.model_root, "j2_plasticity")
        self.assertEqual(model.state_layout.n_state, 7)
        np.testing.assert_array_equal(
            state["plastic_metric_inverse"], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        F = 1.04 * np.eye(3)
        response = update_j2_plasticity(
            MaterialRequest(np.eye(3), F, state, model.properties, None, 0.0, 1.0, True)
        )
        self.assertTrue(response.status.ok, response.status.message)
        self.assertEqual(response.state_trial["equivalent_plastic_strain"], 0.0)
        sigma = response.P @ F.T / np.linalg.det(F)
        np.testing.assert_allclose(sigma, np.trace(sigma) * np.eye(3) / 3.0, atol=2e-11)
        direction = np.array([[0.3, -0.2, 0.1], [0.05, -0.4, 0.07], [0.02, -0.03, 0.25]])
        direction /= np.linalg.norm(direction)
        eps = 2.0e-7
        minus = update_j2_plasticity(
            MaterialRequest(
                np.eye(3), F - eps * direction, state, model.properties, None, 0.0, 1.0, False
            )
        ).P
        plus = update_j2_plasticity(
            MaterialRequest(
                np.eye(3), F + eps * direction, state, model.properties, None, 0.0, 1.0, False
            )
        ).P
        np.testing.assert_allclose(
            np.einsum("iIjJ,jJ->iI", response.A_alg, direction),
            (plus - minus) / (2.0 * eps),
            rtol=2e-7,
            atol=2e-4,
        )

    def test_j2_property_validation(self) -> None:
        properties = self.j2_properties()
        for key in properties:
            missing = properties.copy()
            del missing[key]
            with self.assertRaisesRegex(ModelError, "missing"):
                j2_plasticity_definition("invalid", missing)

        unknown = {**properties, "unused": 1.0}
        with self.assertRaisesRegex(ModelError, "unknown"):
            j2_plasticity_definition("invalid", unknown)

        for key in ("shear_modulus", "bulk_modulus", "initial_yield_stress"):
            invalid = {**properties, key: 0.0}
            with self.assertRaises(ModelError):
                j2_plasticity_definition("invalid", invalid)
        with self.assertRaisesRegex(ModelError, "numeric"):
            j2_plasticity_definition("invalid", {**properties, "saturation_rate": "bad"})
        with self.assertRaisesRegex(ModelError, "finite"):
            j2_plasticity_definition("invalid", {**properties, "saturation_rate": np.inf})

    def test_j2_plastic_return_state_and_tangent(self) -> None:
        model, state = self.j2_initial_state()
        F = np.array([[1.025, 0.012, 0.0], [0.0, 0.988, 0.004], [0.0, 0.0, 0.989]])
        response = update_j2_plasticity(
            MaterialRequest(np.eye(3), F, state, model.properties, None, 0.0, 1.0, True)
        )
        self.assertTrue(response.status.ok, response.status.message)
        ep = float(response.state_trial["equivalent_plastic_strain"])
        self.assertGreater(ep, 0.0)
        packed = np.asarray(response.state_trial["plastic_metric_inverse"])
        Cp_inv = np.array(
            [[packed[0], packed[3], packed[5]],
             [packed[3], packed[1], packed[4]],
             [packed[5], packed[4], packed[2]]]
        )
        self.assertGreater(np.linalg.eigvalsh(Cp_inv)[0], 0.0)
        self.assertAlmostEqual(float(np.linalg.det(Cp_inv)), 1.0, places=12)

        tau = response.P @ F.T
        dev_tau = tau - np.trace(tau) * np.eye(3) / 3.0
        yield_stress = (
            model.properties["initial_yield_stress"]
            + model.properties["linear_hardening_modulus"] * ep
            + model.properties["saturation_increment"]
            * (1.0 - np.exp(-model.properties["saturation_rate"] * ep))
        )
        np.testing.assert_allclose(
            float(np.linalg.norm(dev_tau)), np.sqrt(2.0 / 3.0) * yield_stress,
            rtol=2e-10, atol=1e-7,
        )

        eps = 2.0e-7
        finite_difference = np.empty_like(response.A_alg)
        for j in range(3):
            for J in range(3):
                direction = np.zeros((3, 3))
                direction[j, J] = 1.0
                minus = update_j2_plasticity(
                    MaterialRequest(
                        np.eye(3), F - eps * direction, state,
                        model.properties, None, 0.0, 1.0, False,
                    )
                )
                plus = update_j2_plasticity(
                    MaterialRequest(
                        np.eye(3), F + eps * direction, state,
                        model.properties, None, 0.0, 1.0, False,
                    )
                )
                finite_difference[:, :, j, J] = (plus.P - minus.P) / (2.0 * eps)
        np.testing.assert_allclose(
            response.A_alg, finite_difference, rtol=2e-7, atol=2e-4
        )

    def test_j2_rotation_covariance_and_elastic_unloading(self) -> None:
        model, state0 = self.j2_initial_state()
        F_load = np.diag([1.03, 0.985, 0.985])
        loaded = update_j2_plasticity(
            MaterialRequest(np.eye(3), F_load, state0, model.properties, None, 0.0, 0.5, False)
        )
        self.assertGreater(loaded.state_trial["equivalent_plastic_strain"], 0.0)
        F_unload = np.diag([1.028, 0.986, 0.986])
        unloaded = update_j2_plasticity(
            MaterialRequest(
                F_load, F_unload, loaded.state_trial, model.properties, None, 0.5, 1.0, False
            )
        )
        np.testing.assert_array_equal(unloaded.state_trial.values, loaded.state_trial.values)

        angle = 0.61
        Q = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        base = update_j2_plasticity(
            MaterialRequest(np.eye(3), F_load, state0, model.properties, None, 0.0, 1.0, False)
        )
        rotated = update_j2_plasticity(
            MaterialRequest(np.eye(3), Q @ F_load, state0, model.properties, None, 0.0, 1.0, False)
        )
        np.testing.assert_allclose(rotated.P, Q @ base.P, rtol=2e-13, atol=2e-10)
        np.testing.assert_allclose(rotated.state_trial.values, base.state_trial.values, atol=2e-14)

    def test_j2_trials_do_not_mutate_committed_state(self) -> None:
        model, state0 = self.j2_initial_state()
        first_F = np.diag([1.02, 0.99, 0.99])
        first = update_j2_plasticity(
            MaterialRequest(np.eye(3), first_F, state0, model.properties, None, 0.0, 0.4, False)
        )
        committed = first.state_trial.values.copy()
        update_j2_plasticity(
            MaterialRequest(
                first_F, np.diag([1.20, 0.91, 0.91]), first.state_trial,
                model.properties, None, 0.4, 1.0, False,
            )
        )
        np.testing.assert_array_equal(first.state_trial.values, committed)
        cutback_F = np.diag([1.025, 0.9875, 0.9875])
        after_cutback = update_j2_plasticity(
            MaterialRequest(
                first_F, cutback_F, first.state_trial, model.properties, None, 0.4, 0.5, False
            )
        )
        repeated = update_j2_plasticity(
            MaterialRequest(
                first_F, cutback_F, model.state_layout.view(committed.copy()),
                model.properties, None, 0.4, 0.5, False,
            )
        )
        np.testing.assert_array_equal(after_cutback.P, repeated.P)
        np.testing.assert_array_equal(after_cutback.state_trial.values, repeated.state_trial.values)

    def test_j2_numba_kernel_matches_interpreted_kernel(self) -> None:
        if "numba" not in available_j2_backends():
            self.skipTest("optional Numba backend is unavailable")
        model, state = self.j2_initial_state()
        deformations = (
            np.diag([np.exp(0.001), np.exp(-0.0005), np.exp(-0.0005)]),
            np.array([[1.025, 0.012, 0.0], [0.0, 0.988, 0.004], [0.0, 0.0, 0.989]]),
        )
        try:
            for F in deformations:
                for need_tangent in (False, True):
                    request = MaterialRequest(
                        np.eye(3), F, state, model.properties, None, 0.0, 1.0,
                        need_tangent,
                    )
                    configure_j2_backend("python")
                    interpreted = update_j2_plasticity(request)
                    configure_j2_backend("numba")
                    compiled = update_j2_plasticity(request)
                    self.assertTrue(interpreted.status.ok, interpreted.status.message)
                    self.assertEqual(compiled.status, interpreted.status)
                    np.testing.assert_allclose(
                        compiled.P, interpreted.P, rtol=3.0e-14, atol=3.0e-11
                    )
                    np.testing.assert_allclose(
                        compiled.state_trial.values,
                        interpreted.state_trial.values,
                        rtol=3.0e-14,
                        atol=3.0e-14,
                    )
                    if need_tangent:
                        assert interpreted.A_alg is not None and compiled.A_alg is not None
                        np.testing.assert_allclose(
                            compiled.A_alg, interpreted.A_alg,
                            rtol=3.0e-14, atol=3.0e-11,
                        )
                    else:
                        self.assertIsNone(compiled.A_alg)
            invalid_state_values = state.values.copy()
            invalid_state_values[0] = 2.0
            invalid_state = model.state_layout.view(invalid_state_values)
            failure_requests = (
                MaterialRequest(
                    np.eye(3), np.diag([-1.0, 1.0, 1.0]), state,
                    model.properties, None, 0.0, 1.0, False,
                ),
                MaterialRequest(
                    np.eye(3), np.eye(3), invalid_state,
                    model.properties, None, 0.0, 1.0, False,
                ),
            )
            for request in failure_requests:
                configure_j2_backend("python")
                interpreted = update_j2_plasticity(request)
                configure_j2_backend("numba")
                compiled = update_j2_plasticity(request)
                self.assertFalse(interpreted.status.ok)
                self.assertEqual(compiled.status, interpreted.status)
            self.assertTrue(numba_signatures())
        finally:
            configure_j2_backend("python")

    def test_generic_material_point_driver(self) -> None:
        material = neo_hook_definition("elastic", {"mu": 2.0, "kappa": 12.0})
        target = np.diag([1.1, 1.0 / np.sqrt(1.1), 1.0 / np.sqrt(1.1)])
        history = run_material_path(material, target, 10)
        np.testing.assert_array_equal(history.F[0], np.eye(3))
        np.testing.assert_allclose(history.F[-1], target)
        self.assertEqual(history.state.shape, (11, 0))
        self.assertEqual(history.A_alg.shape, (11, 3, 3, 3, 3))
        self.assertEqual(history.spatial_truesdell_tangent.shape, (11, 6, 6))
        np.testing.assert_allclose(history.material_log_strain[0], 0.0)

    def test_j2_material_point_paths_and_step_refinement(self) -> None:
        material = j2_plasticity_definition("steel", self.j2_properties())

        def uniaxial(parameter: float) -> np.ndarray:
            strain = 0.1 * parameter
            return np.diag([np.exp(strain), np.exp(-0.5 * strain), np.exp(-0.5 * strain)])

        coarse = run_material_path(material, uniaxial, 200, need_tangent=False)
        fine = run_material_path(material, uniaxial, 400)
        ep = fine.state[:, 6]
        self.assertGreater(ep[-1], 0.0)
        self.assertTrue(np.all(np.diff(ep) >= -1.0e-14))
        np.testing.assert_allclose(fine.cauchy_stress[-1], coarse.cauchy_stress[-1], rtol=2e-5)
        np.testing.assert_allclose(fine.state[-1], coarse.state[-1], rtol=2e-4, atol=1e-10)
        D0 = fine.spatial_truesdell_tangent[0]
        mu = self.j2_properties()["shear_modulus"]
        bulk = self.j2_properties()["bulk_modulus"]
        self.assertAlmostEqual(D0[0, 0], bulk + 4.0 * mu / 3.0, places=7)
        self.assertAlmostEqual(D0[0, 1], bulk - 2.0 * mu / 3.0, places=7)
        self.assertAlmostEqual(D0[3, 3], mu, places=7)

        shear = run_material_path(
            material,
            lambda parameter: np.eye(3) + 0.1 * parameter * np.outer([1, 0, 0], [0, 1, 0]),
            400,
            need_tangent=False,
        )
        self.assertGreater(shear.state[-1, 6], 0.0)
        self.assertTrue(np.all(np.diff(shear.state[:, 6]) >= -1.0e-14))


def element_request(formulation: str, u: np.ndarray, need_tangent: bool):
    if formulation == "hex20":
        X = HEX20_PARENT_NODES.copy()
        ngauss = 27
    else:
        X = HEX8_PARENT_NODES.copy()
        ngauss = 8
    return ElementRequest(
        X, np.zeros(X.size), u, np.empty((ngauss, 0)),
        neo_hook_definition("m", {"mu": 1.7, "kappa": 11.0}), None,
        0.0, 0.3, need_tangent, formulation,
    )


def j2_element_request(formulation: str, u: np.ndarray, need_tangent: bool):
    X = HEX8_PARENT_NODES.copy()
    material = j2_plasticity_definition("steel", MaterialTests.j2_properties())
    state0 = init_j2_plasticity(
        MaterialInitRequest(material.properties, None, np.zeros(3), 0.0)
    ).state0
    return ElementRequest(
        X, np.zeros(X.size), u, np.tile(state0, (8, 1)), material, None,
        0.0, 0.3, need_tangent, formulation,
    )


class ElementTests(unittest.TestCase):
    def test_j2_yielded_element_directional_derivatives(self) -> None:
        rng = np.random.default_rng(177)
        H = np.array([[0.025, 0.006, 0.0], [0.0, -0.011, 0.002], [0.0, 0.0, -0.010]])
        for formulation in ("hex8", "hex8_fbar"):
            u = (HEX8_PARENT_NODES @ H.T).ravel() + 1.0e-4 * rng.normal(size=24)
            direction = rng.normal(size=24)
            direction /= np.linalg.norm(direction)
            result = evaluate_element(j2_element_request(formulation, u, True))
            self.assertTrue(result.status.ok, result.status.message)
            self.assertGreater(float(result.state_trial[:, 6].max()), 0.0)
            eps = 1.0e-7
            plus = evaluate_element(j2_element_request(formulation, u + eps * direction, False))
            minus = evaluate_element(j2_element_request(formulation, u - eps * direction, False))
            np.testing.assert_allclose(
                result.K @ direction,
                (plus.f_int - minus.f_int) / (2.0 * eps),
                rtol=2e-6,
                atol=5e-3,
                err_msg=formulation,
            )

    def test_rigid_translation(self) -> None:
        for formulation in ("hex8", "hex8_fbar", "hex20"):
            req = element_request(formulation, np.tile([0.2, -0.1, 0.3], 20 if formulation == "hex20" else 8), True)
            result = evaluate_element(req)
            self.assertTrue(result.status.ok, result.status.message)
            np.testing.assert_allclose(result.f_int, 0.0, atol=2e-13)

    def test_element_directional_derivatives(self) -> None:
        rng = np.random.default_rng(9)
        for formulation in ("hex8", "hex8_fbar", "hex20"):
            X = HEX20_PARENT_NODES if formulation == "hex20" else HEX8_PARENT_NODES
            H = np.array([[0.08, 0.02, -0.01], [0.01, -0.03, 0.015], [0.0, 0.01, 0.04]])
            u = (X @ H.T).ravel() + 0.002 * rng.normal(size=X.size)
            direction = rng.normal(size=X.size)
            direction /= np.linalg.norm(direction)
            result = evaluate_element(element_request(formulation, u, True))
            self.assertTrue(result.status.ok, result.status.message)
            eps = 2e-7
            fp = evaluate_element(element_request(formulation, u + eps * direction, False)).f_int
            fm = evaluate_element(element_request(formulation, u - eps * direction, False)).f_int
            np.testing.assert_allclose(
                result.K @ direction, (fp - fm) / (2 * eps), rtol=2e-7, atol=2e-7,
                err_msg=formulation,
            )

    def test_homogeneous_hex8_and_fbar_agree(self) -> None:
        H = np.diag([0.1, -0.03, 0.02])
        u = (HEX8_PARENT_NODES @ H.T).ravel()
        standard = evaluate_element(element_request("hex8", u, True))
        fbar = evaluate_element(element_request("hex8_fbar", u, True))
        np.testing.assert_allclose(standard.f_int, fbar.f_int, atol=2e-13)
        for a, b in zip(standard.gauss_output.cauchy_stress, fbar.gauss_output.cauchy_stress):
            np.testing.assert_allclose(a, b, atol=2e-13)

    def test_fbar_spatial_and_reference_forms_agree(self) -> None:
        rng = np.random.default_rng(31)
        X = HEX8_PARENT_NODES.copy()
        X[6] += [0.12, -0.07, 0.05]
        u = (X @ np.array([[0.07, 0.02, 0.0], [0.01, -0.02, 0.01], [0.0, 0.01, 0.04]]).T).ravel()
        u += 0.003 * rng.normal(size=24)
        request = ElementRequest(
            X, np.zeros(24), u, np.empty((8, 0)),
            neo_hook_definition("m", {"mu": 1.7, "kappa": 11.0}), None,
            0.0, 0.3, True, "hex8_fbar",
        )
        spatial = evaluate_element(request)
        reference = evaluate_fbar_reference_element(request)
        np.testing.assert_allclose(spatial.f_int, reference.f_int, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(spatial.K, reference.K, rtol=4e-13, atol=4e-13)

    def test_augmented_directional_derivative(self) -> None:
        rng = np.random.default_rng(81)
        X = HEX8_PARENT_NODES.copy()
        u = (X @ np.diag([0.06, -0.01, 0.03])).ravel()
        request = element_request("hex8", u, True)
        response = evaluate_element(request)
        C = sparse.csr_matrix(([1.0, -1.0, 1.0], ([0, 0, 1], [0, 3, 2])), shape=(2, 24))
        lambdas = np.array([0.2, -0.1])
        d = np.array([0.01, -0.03])
        direction = rng.normal(size=26)
        eps = 2e-7

        def residual(displacement, multipliers):
            fint = evaluate_element(element_request("hex8", displacement, False)).f_int
            return np.concatenate([fint + C.T @ multipliers, C @ displacement - d])

        fd = (
            residual(u + eps * direction[:24], lambdas + eps * direction[24:])
            - residual(u - eps * direction[:24], lambdas - eps * direction[24:])
        ) / (2 * eps)
        kkt = sparse.bmat([[response.K, C.T], [C, None]], format="csr")
        np.testing.assert_allclose(kkt @ direction, fd, rtol=2e-7, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
