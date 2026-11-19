"""Regression tier of the patch tests; see docs/PATCH_TEST_PLAN.md.

The engineer-facing tier lives in verification/patch_tests/: frozen,
self-contained simulation cases with their own meshes and decks, run like any
other deck.  This module shares the same construction code
(verification.patch_tests.patch_cube) and keeps the always-on assertions:
consistency of the assembled residual at an exact homogeneous deformation,
machine-precision reproduction through the solver, rigid-body objectivity,
element rank, and traceability of the frozen inputs to the generator.
"""
from __future__ import annotations

import copy
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fe_solver.assembly import assemble_internal, build_model
from fe_solver.config import Deck, load_deck
from fe_solver.elements import evaluate_element
from fe_solver.materials import (
    init_j2_plasticity,
    j2_plasticity_definition,
    neo_hook_definition,
    update_j2_plasticity,
)
from fe_solver.preprocess import prepare_analysis
from fe_solver.quadrature import HEX20_POINTS, HEX8_POINTS
from fe_solver.shape import shape_for
from fe_solver.solver import run_analysis
from fe_solver.types import ElementRequest, MaterialInitRequest, MaterialRequest
from verification.patch_tests.patch_cube import (
    CASES,
    F_FINITE,
    SEEDS,
    analytic_internal_force,
    build_patch_mesh,
    check_frozen_case,
    free_axes,
    interior_dof_mask,
    neo_material_block,
    neo_stress,
    render_deck,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "verification" / "patch_tests"

_J2_PROPERTIES = {
    "shear_modulus": 80193.8,
    "bulk_modulus": 164210.0,
    "initial_yield_stress": 450.0,
    "linear_hardening_modulus": 129.24,
    "saturation_increment": 265.0,
    "saturation_rate": 16.93,
}

_J2_BLOCK = "\n".join(
    ['\n[[materials]]', 'name = "patch"', 'model = "j2_plasticity"', "[materials.properties]"]
    + [f"{key} = {value}" for key, value in _J2_PROPERTIES.items()]
    + [""]
)


def _j2_stress(F: np.ndarray) -> np.ndarray:
    material = j2_plasticity_definition("patch", _J2_PROPERTIES)
    state0 = init_j2_plasticity(
        MaterialInitRequest(material.properties, None, np.zeros(3), 0.0)
    ).state0
    return update_j2_plasticity(
        MaterialRequest(
            np.eye(3), F, material.state_layout.view(state0),
            material.properties, None, 0.0, 1.0, False,
        )
    ).P


def _patch_model(formulation: str, material_block: str, mesh, directory: Path):
    # The deck's mesh file is never opened: the mesh is built in memory and
    # handed to build_model directly, bypassing prepare_analysis/read_gmsh.
    text = render_deck(
        "patch", formulation, "unused-built-in-memory.msh",
        str(directory / "results"), material_block,
    )
    deck_path = directory / f"patch_{formulation}.toml"
    deck_path.write_text(text)
    return build_model(load_deck(deck_path), mesh)


class FrozenInputTests(unittest.TestCase):
    """The committed case inputs must match the generator exactly, so any
    change to a frozen mesh or deck stays traceable through version control."""

    def test_frozen_cases_match_generator(self) -> None:
        for formulation in CASES:
            with self.subTest(formulation=formulation):
                check_frozen_case(CASE_ROOT / formulation, formulation)


class PatchMeshGuardTests(unittest.TestCase):
    """Self-guards: a patch test on an accidentally regular mesh would pass
    trivially (on a regular lattice J0 is diagonal, so e.g. a transposed
    inverse is bit-identical), so the mesh itself is asserted distorted."""

    def test_mesh_is_distorted_valid_and_has_interior_nodes(self) -> None:
        for topology, seed in itertools.product(("hex8", "hex20"), SEEDS):
            with self.subTest(topology=topology, seed=seed):
                mesh, reference = build_patch_mesh(topology, seed)
                free = free_axes(reference)
                offsets = np.linalg.norm(mesh.X - reference, axis=1)
                # Nodes without freedom (cube corners) must not have moved;
                # constrained components stay exactly on their faces/edges.
                np.testing.assert_array_equal(mesh.X[~free], reference[~free])
                # The patch must have free interior nodes, and they must be
                # genuinely off the lattice.
                interior = np.all(free, axis=1)
                self.assertGreater(int(interior.sum()), 0)
                self.assertGreaterEqual(float(offsets[interior].min()), 0.02)
                movable = np.any(free, axis=1)
                self.assertGreaterEqual(float(np.median(offsets[movable])), 0.02)
                # Distortion must not have invalidated the geometry.
                shape = shape_for(mesh.elements[1].topology)
                points = HEX20_POINTS if topology == "hex20" else HEX8_POINTS
                for element in mesh.elements.values():
                    for xi in points:
                        _, dN = shape(xi)
                        detJ0 = float(np.linalg.det(mesh.X[element.connectivity].T @ dN))
                        self.assertGreater(detJ0, 0.0)


class PatchConsistencyTests(unittest.TestCase):
    """PT-1a/PT-2: assemble at the exact homogeneous field, no solve, so the
    assertions are not limited by any Newton stopping tolerance."""

    def assemble_patch(self, formulation, material_block, mesh, F):
        u = ((F - np.eye(3)) @ mesh.X.T).T.ravel()
        with tempfile.TemporaryDirectory() as scratch:
            model = _patch_model(formulation, material_block, mesh, Path(scratch))
        result = assemble_internal(model, np.zeros(mesh.ndof), u, 0.0, 1.0, False)
        outputs = [g for block in result.gauss_output for g in block]
        F_error = max(
            float(np.max(np.abs(Fg - F))) for g in outputs for Fg in g.F_raw
        )
        J_error = max(
            abs(J - float(np.linalg.det(F))) for g in outputs for J in g.J_raw
        )
        stresses = np.asarray([s for g in outputs for s in g.cauchy_stress])
        return result, F_error, J_error, stresses

    def check_homogeneous(self, formulation, material_block, F, P, seeds=SEEDS):
        topology = "hex20" if formulation == "hex20" else "hex8"
        for seed in seeds:
            with self.subTest(formulation=formulation, seed=seed):
                mesh, _ = build_patch_mesh(topology, seed)
                result, F_error, J_error, stresses = self.assemble_patch(
                    formulation, material_block, mesh, F
                )
                interior = interior_dof_mask(mesh)
                self.assertTrue(bool(interior.any()))
                force_scale = float(np.max(np.abs(result.f_int)))
                self.assertLessEqual(
                    float(np.max(np.abs(result.f_int[interior]))),
                    1.0e-11 * force_scale,
                )
                self.assertLessEqual(F_error, 1.0e-12)
                self.assertLessEqual(J_error, 1.0e-12)
                spread = float(np.max(np.ptp(stresses, axis=0)))
                self.assertLessEqual(spread, 1.0e-11 * float(np.max(np.abs(stresses))))
                # The interior-residual identity holds for any *constant*
                # stress, correct or not, so compare the full residual with
                # its independent analytic value as well.
                expected = analytic_internal_force(mesh, topology, P)
                self.assertLessEqual(
                    float(np.max(np.abs(result.f_int - expected))),
                    1.0e-11 * float(np.max(np.abs(expected))),
                )

    def test_hex8_finite_deformation(self) -> None:
        self.check_homogeneous(
            "hex8", neo_material_block(20.0), F_FINITE, neo_stress(F_FINITE, 20.0)
        )

    def test_hex8_fbar_finite_deformation(self) -> None:
        # Homogeneous deformation gives alpha == 1: this certifies F-bar does
        # not break the constant state; the projection itself is verified by
        # test_fbar_spatial_and_reference_forms_agree in test_numerics.py.
        self.check_homogeneous(
            "hex8_fbar", neo_material_block(20.0), F_FINITE, neo_stress(F_FINITE, 20.0)
        )

    def test_hex8_fbar_near_incompressible(self) -> None:
        self.check_homogeneous(
            "hex8_fbar", neo_material_block(1.0e4), F_FINITE,
            neo_stress(F_FINITE, 1.0e4), seeds=(SEEDS[0],),
        )

    def test_hex20_finite_deformation(self) -> None:
        self.check_homogeneous(
            "hex20", neo_material_block(20.0), F_FINITE, neo_stress(F_FINITE, 20.0)
        )

    def test_rigid_motion_produces_no_stress(self) -> None:
        # F = R: zero stress everywhere, so the whole residual (reactions
        # included) must vanish; bounds are absolute, scaled by mu = 1.
        angle = np.pi / 2.0
        R = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        translation = np.array([0.3, -0.2, 0.15])
        for formulation in ("hex8", "hex8_fbar", "hex20"):
            with self.subTest(formulation=formulation):
                mesh, _ = build_patch_mesh(
                    "hex20" if formulation == "hex20" else "hex8", SEEDS[0]
                )
                u = (mesh.X @ R.T + translation - mesh.X).ravel()
                with tempfile.TemporaryDirectory() as scratch:
                    model = _patch_model(
                        formulation, neo_material_block(20.0), mesh, Path(scratch)
                    )
                result = assemble_internal(model, np.zeros(mesh.ndof), u, 0.0, 1.0, False)
                outputs = [g for block in result.gauss_output for g in block]
                F_error = max(
                    float(np.max(np.abs(Fg - R))) for g in outputs for Fg in g.F_raw
                )
                stress = max(
                    float(np.max(np.abs(s))) for g in outputs for s in g.cauchy_stress
                )
                self.assertLessEqual(F_error, 1.0e-12)
                self.assertLessEqual(stress, 1.0e-12)
                self.assertLessEqual(float(np.max(np.abs(result.f_int))), 1.0e-12)

    def test_j2_yielded_patch_state_is_homogeneous(self) -> None:
        F = np.diag([1.05, 0.977, 0.977])
        for formulation in ("hex8", "hex8_fbar"):
            with self.subTest(formulation=formulation):
                mesh, _ = build_patch_mesh("hex8", SEEDS[0])
                result, F_error, J_error, stresses = self.assemble_patch(
                    formulation, _J2_BLOCK, mesh, F
                )
                interior = interior_dof_mask(mesh)
                force_scale = float(np.max(np.abs(result.f_int)))
                self.assertLessEqual(
                    float(np.max(np.abs(result.f_int[interior]))),
                    1.0e-11 * force_scale,
                )
                self.assertLessEqual(F_error, 1.0e-12)
                spread = float(np.max(np.ptp(stresses, axis=0)))
                self.assertLessEqual(spread, 1.0e-11 * float(np.max(np.abs(stresses))))
                expected = analytic_internal_force(mesh, "hex8", _j2_stress(F))
                self.assertLessEqual(
                    float(np.max(np.abs(result.f_int - expected))),
                    1.0e-11 * float(np.max(np.abs(expected))),
                )
                state = result.state_trial[0]
                # Guard: the case must actually yield, and the committed trial
                # state must be homogeneous across the whole patch.
                plastic_strain = state[:, :, 6]
                self.assertGreater(float(plastic_strain.min()), 0.0)
                state_scale = max(1.0, float(np.max(np.abs(state))))
                self.assertLessEqual(
                    float(np.max(np.ptp(state.reshape(-1, state.shape[-1]), axis=0))),
                    1.0e-11 * state_scale,
                )


class SolvedPatchTests(unittest.TestCase):
    """PT-1b: the frozen cases from verification/patch_tests/, loaded through
    the production reader and solved end-to-end; output redirected to a
    temporary directory so the regression run leaves no artifacts."""

    def test_solved_patch_reproduces_the_affine_field(self) -> None:
        for formulation in CASES:
            with self.subTest(formulation=formulation):
                source = load_deck(
                    CASE_ROOT / formulation / f"patch_{formulation}.toml"
                )
                data = copy.deepcopy(source.data)
                with tempfile.TemporaryDirectory() as scratch:
                    data["output"]["directory"] = str(Path(scratch) / "results")
                    prepared = prepare_analysis(Deck(source.path, data, source.curves))
                    result = run_analysis(prepared)
                expected = ((F_FINITE - np.eye(3)) @ prepared.mesh.X.T).T.ravel()
                self.assertLessEqual(
                    float(np.max(np.abs(result.u - expected))), 1.0e-12
                )
                # The deck-driven affine check must agree with the direct one.
                self.assertLessEqual(
                    result.verification["affine_deformation_error_inf"], 1.0e-12
                )
                self.assertEqual(
                    sum(record.cutbacks for record in result.increments), 0
                )
                self.assertLessEqual(
                    max(record.newton_iterations for record in result.increments), 4
                )


class ElementRankTests(unittest.TestCase):
    """A Dirichlet patch test cannot see a spurious zero-energy mode: with all
    boundary nodes prescribed there are too few free DOFs for one to develop,
    at any patch size.  Companion check: the element tangent at the undeformed
    state has exactly six zero modes (translations + infinitesimal rotations)
    and a well-separated seventh eigenvalue."""

    def test_single_distorted_element_has_exactly_six_zero_modes(self) -> None:
        material = neo_hook_definition("m", {"mu": 1.0, "kappa": 20.0})
        for formulation in ("hex8", "hex8_fbar", "hex20"):
            with self.subTest(formulation=formulation):
                mesh, _ = build_patch_mesh(
                    "hex20" if formulation == "hex20" else "hex8", SEEDS[0]
                )
                X = mesh.X[mesh.elements[1].connectivity]
                ngauss = 27 if formulation == "hex20" else 8
                request = ElementRequest(
                    X, np.zeros(X.size), np.zeros(X.size), np.empty((ngauss, 0)),
                    material, None, 0.0, 1.0, True, formulation,
                )
                response = evaluate_element(request)
                self.assertTrue(response.status.ok, response.status.message)
                assert response.K is not None
                # F-bar's tangent is non-symmetric on distorted geometry even
                # at zero stress (B != Bbar): use eigenvalue magnitudes.
                magnitudes = np.sort(np.abs(np.linalg.eigvals(response.K)))
                largest = float(magnitudes[-1])
                zero_modes = int(np.sum(magnitudes < 1.0e-8 * largest))
                self.assertEqual(zero_modes, 6)
                self.assertGreaterEqual(float(magnitudes[6]), 1.0e-5 * largest)


if __name__ == "__main__":
    unittest.main()
