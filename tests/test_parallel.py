from __future__ import annotations

import copy
from contextlib import redirect_stdout
from dataclasses import fields
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fe_solver.assembly import ElementBlock, assemble_internal
from fe_solver.config import Deck, load_deck
from fe_solver.execution import ElementExecutor
from fe_solver.materials import material_definition
from fe_solver.preprocess import prepare_analysis
from fe_solver.solver import run_analysis
from fe_solver.types import GaussOutput, ModelError, RecoverableError
from verification.compare_run_databases import compare_databases


ROOT = Path(__file__).resolve().parents[1]


def _homogeneous_displacement(coordinates: np.ndarray) -> np.ndarray:
    deformation = np.array(
        [
            [1.025, 0.012, 0.0],
            [0.0, 0.985, 0.006],
            [0.0, 0.0, 1.0 / (1.025 * 0.985)],
        ]
    )
    return (coordinates @ (deformation - np.eye(3)).T).ravel()


def _j2_deck(original: Deck) -> Deck:
    data = copy.deepcopy(original.data)
    data["materials"] = [
        {
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
        }
    ]
    data["element_assignments"][0]["material"] = "steel"
    return Deck(original.path, data, original.curves)


def _assert_assemblies_close(
    testcase: unittest.TestCase,
    serial,
    process,
    *,
    rtol: float = 2.0e-14,
    atol: float = 2.0e-14,
) -> None:
    np.testing.assert_allclose(process.f_int, serial.f_int, rtol=rtol, atol=atol)
    assert serial.K is not None and process.K is not None
    np.testing.assert_allclose(
        process.K.toarray(), serial.K.toarray(), rtol=rtol, atol=atol
    )
    for process_state, serial_state in zip(process.state_trial, serial.state_trial):
        np.testing.assert_allclose(process_state, serial_state, rtol=rtol, atol=atol)
    testcase.assertEqual(len(process.gauss_output), len(serial.gauss_output))
    for process_block, serial_block in zip(process.gauss_output, serial.gauss_output):
        testcase.assertEqual(len(process_block), len(serial_block))
        for process_element, serial_element in zip(process_block, serial_block):
            for output_field in fields(GaussOutput):
                np.testing.assert_allclose(
                    np.asarray(getattr(process_element, output_field.name)),
                    np.asarray(getattr(serial_element, output_field.name)),
                    rtol=rtol,
                    atol=atol,
                )


class ElementProcessAssemblyTests(unittest.TestCase):
    def test_process_assembly_matches_serial_for_every_element_formulation(self) -> None:
        cases = (
            ("case_a_hex8.toml", False),
            ("case_a_hex8_fbar.toml", False),
            ("case_a_hex20.toml", False),
            ("case_a_hex8_fbar.toml", True),
        )
        for filename, use_j2 in cases:
            with self.subTest(deck=filename, material="j2" if use_j2 else "neo_hook"):
                original = load_deck(ROOT / "examples" / filename)
                deck = _j2_deck(original) if use_j2 else original
                prepared = prepare_analysis(deck)
                model = prepared.model
                u_n = np.zeros(model.mesh.ndof)
                u_trial = _homogeneous_displacement(model.mesh.X)
                committed_before = [block.state_n.copy() for block in model.blocks]
                serial = assemble_internal(model, u_n, u_trial, 0.0, 0.1, True)
                with ElementExecutor(
                    (block.material for block in model.blocks),
                    sum(len(block.connectivity) for block in model.blocks),
                    2,
                ) as executor:
                    process = assemble_internal(
                        model,
                        u_n,
                        u_trial,
                        0.0,
                        0.1,
                        True,
                        element_executor=executor,
                    )
                _assert_assemblies_close(self, serial, process)
                for block, before in zip(model.blocks, committed_before):
                    np.testing.assert_array_equal(block.state_n, before)

    def test_process_assembly_routes_multiple_material_blocks(self) -> None:
        prepared = prepare_analysis(ROOT / "examples/case_a_hex8_fbar.toml")
        model = prepared.model
        original = model.blocks[0]
        split = len(original.connectivity) // 2
        stiff = material_definition(
            {
                "name": "stiff",
                "model": "neo_hook",
                "properties": {"mu": 10.0, "kappa": 200.0},
            }
        )
        model.blocks = [
            ElementBlock(
                "soft",
                original.formulation,
                original.material,
                original.element_tags[:split],
                original.connectivity[:split],
                original.state_n[:split].copy(),
            ),
            ElementBlock(
                "stiff",
                original.formulation,
                stiff,
                original.element_tags[split:],
                original.connectivity[split:],
                original.state_n[split:].copy(),
            ),
        ]
        u_n = np.zeros(model.mesh.ndof)
        u_trial = _homogeneous_displacement(model.mesh.X)
        serial = assemble_internal(model, u_n, u_trial, 0.0, 0.1, True)
        with ElementExecutor(
            (block.material for block in model.blocks),
            sum(len(block.connectivity) for block in model.blocks),
            2,
        ) as executor:
            process = assemble_internal(
                model,
                u_n,
                u_trial,
                0.0,
                0.1,
                True,
                element_executor=executor,
            )
        _assert_assemblies_close(self, serial, process)

    def test_numba_j2_assembly_matches_python_in_serial_and_process_modes(self) -> None:
        prepared = prepare_analysis(_j2_deck(load_deck(ROOT / "examples/case_a_hex8_fbar.toml")))
        model = prepared.model
        u_n = np.zeros(model.mesh.ndof)
        u_trial = _homogeneous_displacement(model.mesh.X)
        materials = tuple(block.material for block in model.blocks)
        element_count = sum(len(block.connectivity) for block in model.blocks)
        with ElementExecutor(
            materials, element_count, 1, j2_backend="python"
        ) as executor:
            python = assemble_internal(
                model, u_n, u_trial, 0.0, 0.1, True, element_executor=executor
            )
        with ElementExecutor(
            materials, element_count, 1, j2_backend="numba"
        ) as executor:
            numba_serial = assemble_internal(
                model, u_n, u_trial, 0.0, 0.1, True, element_executor=executor
            )
        with ElementExecutor(
            materials, element_count, 2, j2_backend="numba"
        ) as executor:
            numba_process = assemble_internal(
                model, u_n, u_trial, 0.0, 0.1, True, element_executor=executor
            )
        _assert_assemblies_close(self, python, numba_serial, rtol=1.0e-12, atol=1.0e-10)
        _assert_assemblies_close(self, python, numba_process, rtol=1.0e-12, atol=1.0e-10)

    def test_persistent_process_solver_matches_serial_and_logs_timing(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        with tempfile.TemporaryDirectory(prefix="fe-parallel-test-") as directory:
            root = Path(directory)
            results = []
            for name, processes in (("serial", 1), ("process", 2)):
                data = copy.deepcopy(original.data)
                data["output"]["directory"] = str(root / name)
                data["restart"]["enabled"] = False
                deck = Deck(original.path, data, original.curves)
                with redirect_stdout(StringIO()):
                    results.append(
                        run_analysis(
                            deck,
                            stop_time=0.05,
                            num_processes=processes,
                            debug_timing=True,
                        )
                    )

            serial, process = results
            np.testing.assert_allclose(process.u, serial.u, rtol=2.0e-14, atol=2.0e-14)
            np.testing.assert_allclose(
                process.lambdas, serial.lambdas, rtol=2.0e-14, atol=2.0e-14
            )
            self.assertEqual(process.execution["element_backend"], "process")
            self.assertEqual(process.execution["requested_processes"], 2)
            self.assertEqual(process.execution["effective_processes"], 2)
            for process_block, serial_block in zip(process.model.blocks, serial.model.blocks):
                np.testing.assert_allclose(
                    process_block.state_n, serial_block.state_n, rtol=2.0e-14, atol=2.0e-14
                )

            comparison = compare_databases(
                root / "serial/run.h5",
                root / "process/run.h5",
                rtol=2.0e-14,
                atol=2.0e-14,
            )
            self.assertTrue(comparison.passed, comparison.structural_errors)

            with (root / "process/run_log.json").open(encoding="utf-8") as stream:
                log = json.load(stream)
            self.assertEqual(log["schema_version"], 2)
            self.assertEqual(log["execution"], process.execution)
            self.assertGreater(log["timing"]["elapsed_wall_seconds"], 0.0)
            self.assertTrue(log["newton_history"])
            for record in log["newton_history"]:
                for name in (
                    "iteration_wall_seconds",
                    "assembly_wall_seconds",
                    "line_search_assembly_wall_seconds",
                    "kkt_factorization_wall_seconds",
                    "kkt_solve_wall_seconds",
                    "element_phase_wall_seconds",
                    "sparse_finalize_wall_seconds",
                ):
                    self.assertIn(name, record)
                    self.assertGreaterEqual(record[name], 0.0)

    def test_recoverable_element_failure_drains_pool_and_preserves_committed_state(self) -> None:
        prepared = prepare_analysis(ROOT / "examples/case_a_hex8.toml")
        model = prepared.model
        u_n = np.zeros(model.mesh.ndof)
        inverted = (model.mesh.X @ (np.diag([-1.0, 1.0, 1.0]) - np.eye(3)).T).ravel()
        committed_before = [block.state_n.copy() for block in model.blocks]
        with ElementExecutor(
            (block.material for block in model.blocks),
            sum(len(block.connectivity) for block in model.blocks),
            2,
        ) as executor:
            with self.assertRaises(RecoverableError):
                assemble_internal(
                    model,
                    u_n,
                    inverted,
                    0.0,
                    0.1,
                    True,
                    element_executor=executor,
                )
            recovered = assemble_internal(
                model,
                u_n,
                u_n,
                0.0,
                0.05,
                True,
                element_executor=executor,
            )
        self.assertTrue(np.all(np.isfinite(recovered.f_int)))
        for block, before in zip(model.blocks, committed_before):
            np.testing.assert_array_equal(block.state_n, before)

    def test_process_count_validation(self) -> None:
        original = load_deck(ROOT / "examples/case_a_hex8.toml")
        model = prepare_analysis(original).model
        materials = [block.material for block in model.blocks]
        for invalid in (True, 0, -1, 1.0, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ModelError, "positive integer"):
                    ElementExecutor(materials, 1, invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
