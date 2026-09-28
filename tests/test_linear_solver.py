from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from fe_solver.config import Deck, load_deck
from fe_solver.linear_solver import (
    SUPERLU_MT_LIBRARY,
    LinearSolverConfig,
    close_factor,
    factor_kkt,
    solve_factor,
)
from fe_solver.solver import run_analysis
from fe_solver.types import ModelError


ROOT = Path(__file__).resolve().parents[1]


class LinearSolverTests(unittest.TestCase):
    def test_superlu_mt_requires_explicit_positive_thread_count(self) -> None:
        with self.assertRaisesRegex(ModelError, "explicit positive"):
            LinearSolverConfig("superlu_mt")
        with self.assertRaisesRegex(ModelError, "explicit positive"):
            LinearSolverConfig("superlu_mt", 0)
        with self.assertRaisesRegex(ModelError, "only with superlu_mt"):
            LinearSolverConfig("scipy_splu", 2)
        self.assertEqual(
            LinearSolverConfig("superlu_mt", 3).num_threads,
            3,
        )

    def test_superlu_mt_selection_is_lazy_and_forwards_thread_count(self) -> None:
        K = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
        C = sparse.csr_matrix([[1.0, -1.0]])
        sentinel = object()
        with (
            mock.patch(
                "fe_solver.linear_solver._load_superlu_mt",
                return_value="library",
            ) as load,
            mock.patch(
                "native.superlu_mt_backend.SuperLUMTFactor.factor",
                return_value=sentinel,
            ) as factor,
        ):
            result = factor_kkt(
                K,
                C,
                "COLAMD",
                LinearSolverConfig("superlu_mt", 3),
            )
        self.assertIs(result, sentinel)
        load.assert_called_once_with()
        self.assertEqual(factor.call_args.kwargs["nprocs"], 3)
        self.assertEqual(factor.call_args.kwargs["ordering"], "COLAMD")

    def test_scipy_factorization_does_not_load_superlu_mt(self) -> None:
        K = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
        C = sparse.csr_matrix([[1.0, -1.0]])
        with mock.patch(
            "fe_solver.linear_solver._load_superlu_mt",
            side_effect=AssertionError("native library was loaded"),
        ):
            factor = factor_kkt(K, C, "COLAMD")
            solution = factor.solve(np.array([1.0, 2.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(solution)))

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(),
        "SuperLU_MT bridge has not been built",
    )
    def test_built_superlu_mt_bridge_matches_scipy(self) -> None:
        K = sparse.csr_matrix(
            [[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.0, 2.0, 2.0]]
        )
        C = sparse.csr_matrix([[1.0, 0.0, -1.0]])
        rhs = np.array([1.0, -2.0, 0.5, 0.3])
        scipy_factor = factor_kkt(K, C, "COLAMD")
        config = LinearSolverConfig("superlu_mt", 1)
        mt_factor = factor_kkt(K, C, "COLAMD", config)
        try:
            expected = scipy_factor.solve(rhs)
            actual = solve_factor(mt_factor, rhs, config)
            np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-11)
        finally:
            close_factor(mt_factor)

    def test_newton_attempt_owns_exact_and_modified_factors(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")

        class TrackingFactor:
            def __init__(self, factor):
                self.factor = factor
                self.close_count = 0

            def solve(self, rhs):
                return self.factor.solve(rhs)

            def close(self):
                self.close_count += 1

        with tempfile.TemporaryDirectory(prefix="fe-linear-solver-test-") as directory:
            for method in ("newton", "modified_newton"):
                with self.subTest(method=method):
                    factors = []

                    def tracking_factor(K, C, ordering, config):
                        factor = TrackingFactor(factor_kkt(K, C, ordering, config))
                        factors.append(factor)
                        return factor

                    data = copy.deepcopy(original.data)
                    data["nonlinear"]["method"] = method
                    data["output"]["directory"] = str(Path(directory) / method)
                    data["restart"]["enabled"] = False
                    deck = Deck(original.path, data, original.curves)
                    with mock.patch(
                        "fe_solver.solver._factor_kkt",
                        side_effect=tracking_factor,
                    ):
                        result = run_analysis(deck, stop_time=0.05)

                    self.assertTrue(factors)
                    self.assertTrue(
                        all(factor.close_count == 1 for factor in factors)
                    )
                    if method == "modified_newton":
                        self.assertEqual(
                            len(factors),
                            sum(increment.attempts for increment in result.increments),
                        )

    def test_run_analysis_routes_selected_backend_and_threads(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        received = []

        def scipy_stand_in(K, C, ordering, config):
            received.append(config)
            return factor_kkt(K, C, ordering)

        with tempfile.TemporaryDirectory(prefix="fe-linear-solver-test-") as directory:
            data = copy.deepcopy(original.data)
            data["output"]["directory"] = directory
            data["restart"]["enabled"] = False
            deck = Deck(original.path, data, original.curves)
            with mock.patch(
                "fe_solver.solver._factor_kkt",
                side_effect=scipy_stand_in,
            ):
                result = run_analysis(
                    deck,
                    stop_time=0.05,
                    solver_backend="superlu_mt",
                    num_threads=3,
                )

        self.assertTrue(received)
        self.assertTrue(
            all(config == LinearSolverConfig("superlu_mt", 3) for config in received)
        )
        self.assertEqual(result.analysis["linear_system"]["backend"], "superlu_mt")
        self.assertEqual(
            result.analysis["linear_system"]["options"]["num_threads"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
