from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from fe_solver.elements import evaluate_element, evaluate_fbar_reference_element
from fe_solver.materials import evaluate_neo_hookean, neo_hookean_definition
from fe_solver.shape import HEX20_PARENT_NODES, HEX8_PARENT_NODES, hex20_shape, hex8_shape
from fe_solver.types import ElementRequest, MaterialRequest


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
    def test_neo_hookean_tangent(self) -> None:
        rng = np.random.default_rng(7)
        F = np.eye(3) + 0.15 * rng.normal(size=(3, 3))
        params = {"mu": 2.3, "kappa": 17.0}
        req = MaterialRequest(np.eye(3), F, np.empty(0), params, None, 0.0, 1.0, True)
        result = evaluate_neo_hookean(req)
        self.assertTrue(result.status.ok)
        dF = rng.normal(size=(3, 3))
        dF /= np.linalg.norm(dF)
        eps = 2e-7
        pm = evaluate_neo_hookean(
            MaterialRequest(np.eye(3), F - eps * dF, np.empty(0), params, None, 0, 1, False)
        ).P
        pp = evaluate_neo_hookean(
            MaterialRequest(np.eye(3), F + eps * dF, np.empty(0), params, None, 0, 1, False)
        ).P
        analytic = np.einsum("iIjJ,jJ->iI", result.A_alg, dF)
        np.testing.assert_allclose(analytic, (pp - pm) / (2 * eps), rtol=2e-8, atol=2e-8)
        no_tangent = evaluate_neo_hookean(
            MaterialRequest(np.eye(3), F, np.empty(0), params, None, 0, 1, False)
        )
        np.testing.assert_array_equal(result.P, no_tangent.P)
        self.assertIsNone(no_tangent.A_alg)


def element_request(formulation: str, u: np.ndarray, need_tangent: bool):
    if formulation == "hex20":
        X = HEX20_PARENT_NODES.copy()
        ngauss = 27
    else:
        X = HEX8_PARENT_NODES.copy()
        ngauss = 8
    return ElementRequest(
        X, np.zeros(X.size), u, np.empty((ngauss, 0)),
        neo_hookean_definition("m", {"mu": 1.7, "kappa": 11.0}), None,
        0.0, 0.3, need_tangent, formulation,
    )


class ElementTests(unittest.TestCase):
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
            neo_hookean_definition("m", {"mu": 1.7, "kappa": 11.0}), None,
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
