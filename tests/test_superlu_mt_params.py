"""Contract tests for the expert-level SuperLU_MT parameter layer.

Every test here is pure Python or uses a single tiny matrix: none of them
factorize anything large and none of them raise the thread count, so the suite
stays within the project's testing core budget (see AGENTS.md).
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scipy import sparse

from fe_solver.config import Deck, load_deck
from fe_solver.linear_solver import (
    SUPERLU_MT_LIBRARY,
    LinearSolverConfig,
    factor_kkt,
)
from fe_solver.preprocess import prepare_analysis
from fe_solver.solver import run_analysis
from fe_solver.superlu_mt_params import (
    DEFAULTS,
    SuperLUMTParams,
    format_override_banner,
    resolve_params,
)
from fe_solver.types import ModelError


ROOT = Path(__file__).resolve().parents[1]


def write_params(body: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".toml", delete=False, encoding="utf-8"
    )
    with handle:
        handle.write(body)
    return Path(handle.name)


class ResolutionTests(unittest.TestCase):
    def test_defaults_apply_when_nothing_opts_in(self) -> None:
        params = resolve_params()
        self.assertEqual(params.factor_kwargs(), DEFAULTS)
        self.assertEqual(set(params.sources.values()), {"default"})
        self.assertEqual(params.overrides, {})
        self.assertIsNone(params.params_file)

    def test_file_overrides_defaults_and_deck_overrides_file(self) -> None:
        path = write_params(
            "[superlu_mt]\ndiag_pivot_thresh = 0.5\npanel_size = 12\n"
        )
        try:
            params = resolve_params({"panel_size": 20, "relax": 6}, path)
        finally:
            path.unlink()
        # The file sets diag_pivot_thresh and panel_size; the deck re-sets
        # panel_size and adds relax, so the deck must win on panel_size.
        self.assertEqual(params.diag_pivot_thresh, 0.5)
        self.assertEqual(params.panel_size, 20)
        self.assertEqual(params.relax, 6)
        self.assertEqual(
            {k: v for k, v in params.sources.items() if v != "default"},
            {"diag_pivot_thresh": "file", "panel_size": "deck", "relax": "deck"},
        )
        self.assertEqual(sorted(params.overrides), ["diag_pivot_thresh", "panel_size", "relax"])

    def test_params_file_is_recorded_with_its_digest(self) -> None:
        path = write_params("[superlu_mt]\nrelax = 6\n")
        try:
            params = resolve_params(None, path)
        finally:
            path.unlink()
        self.assertEqual(params.params_file, path.resolve())
        self.assertEqual(len(params.params_file_sha256), 64)
        self.assertEqual(params.summary_options()["params_file"], str(path.resolve()))


class StrictValidationTests(unittest.TestCase):
    def test_unknown_deck_parameter_is_an_error(self) -> None:
        with self.assertRaisesRegex(ModelError, "unknown SuperLU_MT parameters"):
            resolve_params({"panel_sixe": 8})

    def test_unknown_file_parameter_is_an_error(self) -> None:
        path = write_params("[superlu_mt]\ndrop_tol = 0.1\n")
        try:
            with self.assertRaisesRegex(ModelError, "unknown SuperLU_MT parameters"):
                resolve_params(None, path)
        finally:
            path.unlink()

    def test_unexpected_table_in_file_is_an_error(self) -> None:
        path = write_params("[solver]\nrelax = 6\n")
        try:
            with self.assertRaisesRegex(ModelError, "unexpected tables"):
                resolve_params(None, path)
        finally:
            path.unlink()

    def test_missing_file_is_an_error(self) -> None:
        with self.assertRaisesRegex(ModelError, "cannot read SuperLU_MT parameter file"):
            resolve_params(None, "/nonexistent/superlu_mt_params.toml")

    def test_out_of_range_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ModelError, r"diag_pivot_thresh must be a finite value"):
            resolve_params({"diag_pivot_thresh": 1.5})
        with self.assertRaisesRegex(ModelError, "panel_size must be -1"):
            resolve_params({"panel_size": 0})
        with self.assertRaisesRegex(ModelError, "relax must be -1"):
            resolve_params({"relax": -2})
        with self.assertRaisesRegex(ModelError, "must be an integer"):
            resolve_params({"panel_size": 8.5})
        with self.assertRaisesRegex(ModelError, "must be a number, got a boolean"):
            resolve_params({"panel_size": True})

    def test_blocking_parameters_reject_zero_and_below_minus_one(self) -> None:
        for name in ("max_supernode", "rowblk", "colblk"):
            with self.assertRaisesRegex(ModelError, f"{name} must be -1"):
                resolve_params({name: 0})
            with self.assertRaisesRegex(ModelError, f"{name} must be -1"):
                resolve_params({name: -2})
            self.assertEqual(getattr(resolve_params({name: 64}), name), 64)

    def test_fill_parameters_accept_both_semantics_but_not_overflow(self) -> None:
        params = resolve_params(
            {"fill_lusup": -100, "fill_ucol": 5_000_000, "fill_lsub": 0}
        )
        self.assertEqual(params.fill_lusup, -100)
        self.assertEqual(params.fill_ucol, 5_000_000)
        self.assertEqual(params.fill_lsub, 0)
        self.assertNotIn("fill_lsub", params.overrides)
        with self.assertRaisesRegex(ModelError, "must fit a 32-bit integer"):
            resolve_params({"fill_lusup": 2**31})
        with self.assertRaisesRegex(ModelError, "must be an integer"):
            resolve_params({"fill_ucol": -50.5})


class VisibilityTests(unittest.TestCase):
    def test_default_run_prints_no_banner(self) -> None:
        self.assertEqual(format_override_banner(resolve_params()), [])

    def test_banner_names_every_override_with_its_default(self) -> None:
        banner = "\n".join(
            format_override_banner(resolve_params({"panel_size": 12}))
        )
        self.assertIn(
            f"EXPERT SuperLU_MT OVERRIDES ACTIVE (1 of {len(DEFAULTS)} parameters)",
            banner,
        )
        self.assertIn("panel_size = 12  <-- OVERRIDDEN by deck (default -1)", banner)
        self.assertIn("diag_pivot_thresh = 1.0 (default)", banner)

    def test_relaxed_pivoting_is_called_out(self) -> None:
        banner = "\n".join(
            format_override_banner(resolve_params({"diag_pivot_thresh": 0.1}))
        )
        self.assertIn("WARNING: diag_pivot_thresh < 1.0", banner)

    def test_absolute_fill_counts_are_called_out(self) -> None:
        banner = "\n".join(
            format_override_banner(
                resolve_params({"fill_lusup": 1_000_000, "fill_ucol": -100})
            )
        )
        self.assertIn("fill_lusup = 1000000  <-- OVERRIDDEN by deck (default 0)", banner)
        self.assertIn("NOTE: positive fill_lusup are ABSOLUTE entry", banner)
        self.assertNotIn("fill_ucol are ABSOLUTE", banner)

    def test_loaded_file_is_announced_even_without_overrides(self) -> None:
        path = write_params("[superlu_mt]\nrelax = -1\n")
        try:
            banner = "\n".join(format_override_banner(resolve_params(None, path)))
        finally:
            path.unlink()
        self.assertIn("no parameter differs from defaults", banner)
        self.assertIn(str(path.resolve()), banner)


class ForwardingTests(unittest.TestCase):
    def test_parameters_reach_the_native_backend(self) -> None:
        K = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
        C = sparse.csr_matrix([[1.0, -1.0]])
        params = resolve_params({"diag_pivot_thresh": 0.25, "panel_size": 12, "relax": 6})
        with (
            mock.patch(
                "fe_solver.linear_solver._load_superlu_mt", return_value="library"
            ),
            mock.patch(
                "native.superlu_mt_backend.SuperLUMTFactor.factor", return_value=object()
            ) as factor,
        ):
            factor_kkt(K, C, "COLAMD", LinearSolverConfig("superlu_mt", 2, params))
        self.assertEqual(factor.call_args.kwargs["diag_pivot_thresh"], 0.25)
        self.assertEqual(factor.call_args.kwargs["panel_size"], 12)
        self.assertEqual(factor.call_args.kwargs["relax"], 6)

    def test_defaults_are_forwarded_when_no_parameters_are_given(self) -> None:
        K = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
        C = sparse.csr_matrix([[1.0, -1.0]])
        with (
            mock.patch(
                "fe_solver.linear_solver._load_superlu_mt", return_value="library"
            ),
            mock.patch(
                "native.superlu_mt_backend.SuperLUMTFactor.factor", return_value=object()
            ) as factor,
        ):
            factor_kkt(K, C, "COLAMD", LinearSolverConfig("superlu_mt", 2))
        for name, value in DEFAULTS.items():
            self.assertEqual(factor.call_args.kwargs[name], value)

    def test_fill_parameters_reach_the_native_backend(self) -> None:
        K = sparse.csr_matrix([[3.0, 1.0], [2.0, 4.0]])
        C = sparse.csr_matrix([[1.0, -1.0]])
        params = resolve_params({"fill_lusup": -100, "fill_ucol": -80, "fill_lsub": -60})
        with (
            mock.patch(
                "fe_solver.linear_solver._load_superlu_mt", return_value="library"
            ),
            mock.patch(
                "native.superlu_mt_backend.SuperLUMTFactor.factor", return_value=object()
            ) as factor,
        ):
            factor_kkt(K, C, "COLAMD", LinearSolverConfig("superlu_mt", 2, params))
        self.assertEqual(factor.call_args.kwargs["fill_lusup"], -100)
        self.assertEqual(factor.call_args.kwargs["fill_ucol"], -80)
        self.assertEqual(factor.call_args.kwargs["fill_lsub"], -60)

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(), "SuperLU_MT bridge has not been built"
    )
    def test_fill_overrides_still_solve_correctly_through_the_bridge(self) -> None:
        import numpy as np

        from fe_solver.linear_solver import close_factor, solve_factor

        K = sparse.csr_matrix(
            [[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.0, 2.0, 2.0]]
        )
        C = sparse.csr_matrix([[1.0, 0.0, -1.0]])
        rhs = np.array([1.0, -2.0, 0.5, 0.3])
        expected = factor_kkt(K, C, "COLAMD").solve(rhs)
        # Both semantics: growth factors and generous absolute entry counts.
        for fills in (
            {"fill_lusup": -100, "fill_ucol": -80, "fill_lsub": -60},
            {"fill_lusup": 100_000, "fill_ucol": 100_000, "fill_lsub": 100_000},
        ):
            config = LinearSolverConfig("superlu_mt", 1, resolve_params(fills))
            mt_factor = factor_kkt(K, C, "COLAMD", config)
            try:
                actual = solve_factor(mt_factor, rhs, config)
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
            finally:
                close_factor(mt_factor)

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(), "SuperLU_MT bridge has not been built"
    )
    def test_factor_diagnostics_are_captured_and_survive_close(self) -> None:
        K = sparse.csr_matrix(
            [[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.0, 2.0, 2.0]]
        )
        C = sparse.csr_matrix([[1.0, 0.0, -1.0]])
        from fe_solver.linear_solver import close_factor

        config = LinearSolverConfig("superlu_mt", 1)
        mt_factor = factor_kkt(K, C, "COLAMD", config)
        close_factor(mt_factor)
        diag = mt_factor.diagnostics
        self.assertIsNotNone(diag)
        kkt_order = K.shape[0] + C.shape[0]
        # The factors hold at least one entry per row/column each.
        self.assertGreaterEqual(diag["nnz_L"], kkt_order)
        self.assertGreaterEqual(diag["nnz_U"], kkt_order)
        self.assertGreater(diag["factor_flops"], 0.0)
        self.assertGreater(diag["lu_storage_bytes"], 0.0)
        self.assertGreaterEqual(
            diag["total_needed_bytes"], diag["lu_storage_bytes"]
        )
        self.assertGreaterEqual(diag["expansions"], 0.0)

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(), "SuperLU_MT bridge has not been built"
    )
    def test_bridge_sp_ienv_override_matches_upstream_defaults(self) -> None:
        """The bridge replaces sp_ienv(), so its fallbacks must track upstream.

        These are the generic (non-MACH) values in upstream SRC/sp_ienv.c. If
        upstream changes them, or the build starts defining MACH, this fails
        and the override in native/superlu_mt.c must be revisited.
        """
        import ctypes

        from native.superlu_mt_backend import load_superlu_mt

        lib = load_superlu_mt(SUPERLU_MT_LIBRARY)
        lib.slumt_sp_ienv_defaults.restype = ctypes.c_int
        lib.slumt_sp_ienv_defaults.argtypes = [ctypes.c_int]
        upstream = {1: 20, 2: 6, 3: 200, 4: 200, 5: 100, 6: -50, 7: -50, 8: -30}
        for ispec, expected in upstream.items():
            self.assertEqual(lib.slumt_sp_ienv_defaults(ispec), expected, f"sp_ienv({ispec})")

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(), "SuperLU_MT bridge has not been built"
    )
    def test_library_internals_resolve_to_the_bridge_sp_ienv(self) -> None:
        """Proves the link-time override actually reaches SuperLU_MT's internals."""
        import ctypes

        from fe_solver.linear_solver import close_factor
        from native.superlu_mt_backend import load_superlu_mt

        lib = load_superlu_mt(SUPERLU_MT_LIBRARY)
        lib.slumt_sp_ienv_call_count.restype = ctypes.c_long
        before = lib.slumt_sp_ienv_call_count()
        K = sparse.csr_matrix([[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.0, 2.0, 2.0]])
        C = sparse.csr_matrix([[1.0, 0.0, -1.0]])
        close_factor(factor_kkt(K, C, "COLAMD", LinearSolverConfig("superlu_mt", 1)))
        # The bridge itself calls sp_ienv at most twice (panel_size, relax);
        # anything beyond that came from inside the library.
        self.assertGreater(lib.slumt_sp_ienv_call_count() - before, 2)

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(), "SuperLU_MT bridge has not been built"
    )
    def test_blocking_overrides_do_not_change_the_solution(self) -> None:
        import numpy as np

        from fe_solver.linear_solver import close_factor, solve_factor

        K = sparse.csr_matrix([[3.0, 2.0, 0.0], [-1.0, 4.0, 1.0], [0.0, 2.0, 2.0]])
        C = sparse.csr_matrix([[1.0, 0.0, -1.0]])
        rhs = np.array([1.0, -2.0, 0.5, 0.3])
        expected = factor_kkt(K, C, "COLAMD").solve(rhs)
        for blocking in (
            {"max_supernode": 1},
            {"max_supernode": 4, "rowblk": 8, "colblk": 4},
            {"rowblk": 1000, "colblk": 1000},
        ):
            config = LinearSolverConfig("superlu_mt", 1, resolve_params(blocking))
            factor = factor_kkt(K, C, "COLAMD", config)
            try:
                np.testing.assert_allclose(
                    solve_factor(factor, rhs, config), expected, rtol=1e-9, atol=1e-11
                )
            finally:
                close_factor(factor)

    def test_parameters_are_rejected_for_other_backends(self) -> None:
        with self.assertRaisesRegex(ModelError, "only with the superlu_mt backend"):
            LinearSolverConfig("scipy_splu", None, SuperLUMTParams())


class DeckIntegrationTests(unittest.TestCase):
    def deck_with(self, linear_solver: dict[str, object]) -> Deck:
        deck = load_deck(ROOT / "examples/case_a_hex8.toml")
        data = copy.deepcopy(deck.data)
        data["linear_solver"].update(linear_solver)
        return Deck(deck.path, data, deck.curves)

    def test_parameters_in_a_deck_are_rejected_for_other_backends(self) -> None:
        deck = self.deck_with({"backend": "scipy_splu", "superlu_mt": {"relax": 6}})
        with self.assertRaisesRegex(ModelError, r"apply only to superlu_mt"):
            run_analysis(prepare_analysis(deck), stop_time=0.05)

    @unittest.skipUnless(
        SUPERLU_MT_LIBRARY.is_file(), "SuperLU_MT bridge has not been built"
    )
    def test_run_summary_records_effective_values_and_provenance(self) -> None:
        deck = self.deck_with(
            {"backend": "superlu_mt", "superlu_mt": {"panel_size": 12}}
        )
        # Four threads keeps this inside the project's testing core budget.
        result = run_analysis(
            prepare_analysis(deck), stop_time=0.05, num_threads=4
        )
        recorded = result.analysis["linear_system"]["options"]["superlu_mt"]
        self.assertEqual(recorded["values"]["panel_size"], 12)
        self.assertEqual(recorded["sources"]["panel_size"], "deck")
        self.assertEqual(recorded["sources"]["relax"], "default")
        self.assertEqual(recorded["overridden"], ["panel_size"])
        self.assertEqual(recorded["defaults"], DEFAULTS)
        diagnostics = result.analysis["linear_system"]["factor_diagnostics"]
        self.assertGreater(diagnostics["nnz_L"], 0.0)
        self.assertGreater(diagnostics["factor_flops"], 0.0)
        self.assertGreaterEqual(diagnostics["expansions"], 0)


if __name__ == "__main__":
    unittest.main()
