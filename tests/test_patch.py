"""Patch tests on distorted 2x2x2 element patches; see docs/PATCH_TEST_PLAN.md.

A homogeneous finite deformation u = (Fbar - I) X is imposed on every node of a
distorted patch and must be reproduced to machine precision: constant F at all
Gauss points, constant Cauchy stress, zero residual at interior nodes.  This
certifies the residual chain (shape functions, kinematics, B, stress transform,
quadrature volume, assembly) on geometry where a regular mesh would hide
errors.  It does not certify the tangent, which the directional-derivative
tests in test_numerics.py cover.
"""
from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fe_solver.assembly import assemble_internal, build_model
from fe_solver.config import load_deck, mandatory_events
from fe_solver.constraints import build_constraints
from fe_solver.elements import evaluate_element
from fe_solver.materials import (
    init_j2_plasticity,
    j2_plasticity_definition,
    neo_hook_definition,
    update_j2_plasticity,
    update_neo_hook,
)
from fe_solver.mesh import Mesh, MeshElement
from fe_solver.preprocess import PreparedAnalysis
from fe_solver.shape import HEX20_PARENT_NODES, hex8_shape, shape_for
from fe_solver.quadrature import HEX20_POINTS, HEX20_WEIGHTS, HEX8_POINTS, HEX8_WEIGHTS
from fe_solver.solver import run_analysis
from fe_solver.types import (
    ElementRequest,
    MaterialInitRequest,
    MaterialRequest,
    StateLayout,
)


_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
_SEEDS = (11, 29)
_DISTORTION = 0.18

# Finite, non-symmetric (a transpose slip cannot pass), with shear and volume
# change; det = 1.1501.
_F_FINITE = np.array(
    [[1.20, 0.15, -0.08],
     [0.05, 0.92, 0.10],
     [-0.04, 0.07, 1.06]]
)

_NEO_BLOCK = """
[[materials]]
name = "patch"
model = "neo_hook"
[materials.properties]
mu = 1.0
kappa = @KAPPA@
"""

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

_DECK_TEMPLATE = """
[analysis]
name = "patch"
t_start = 0.0
t_end = 1.0

# load_deck requires a [mesh] table, but this file is never opened: the patch
# mesh is built in memory by build_patch_mesh() and handed to build_model()
# and PreparedAnalysis directly, bypassing prepare_analysis()/read_gmsh().
[mesh]
file = "built-in-memory-see-build_patch_mesh.msh"
format = "gmsh_msh41"

[time]
dt_initial = 0.25
dt_min = 1.0e-5
dt_max = 0.25
cutback_factor = 0.5
growth_factor = 1.25
grow_if_newton_iterations_le = 4
max_attempts_per_increment = 8

[nonlinear]
method = "newton"
max_iterations = 12
force_rtol = 1.0e-12
force_atol = 1.0e-14
constraint_rtol = 1.0e-14
constraint_atol = 1.0e-14

[linear_solver]
backend = "scipy_splu"
ordering = "COLAMD"
singular_policy = "diagnose_and_abort"
@MATERIAL@
[[element_assignments]]
region = "solid"
formulation = "@FORMULATION@"
material = "patch"

[output]
directory = "@OUTPUT@"
database = "run.h5"
history_file = "run_log.json"
explicit_times = [1.0]

[restart]
enabled = false
keep_last = 0
restart_from = ""
"""

_VERIFICATION_BLOCK = """
[verification]
require_positive_J = true
check_global_force_balance = true
affine_deformation = true
affine_tolerance = 1.0e-12
"""


def _free_axes(reference: np.ndarray) -> np.ndarray:
    """Directions a node may move without leaving its cube faces/edges."""
    return (reference > 1.0e-12) & (reference < 1.0 - 1.0e-12)


def build_patch_mesh(
    topology: str, seed: int, distortion: float = _DISTORTION
) -> tuple[Mesh, np.ndarray]:
    """Distorted 2x2x2 unit-cube patch; returns the mesh and the undistorted
    reference lattice used for node classification."""
    rng = np.random.default_rng(seed)
    n = 2
    h = 1.0 / n
    grid = np.linspace(0.0, 1.0, n + 1)
    index: dict[tuple[int, int, int], int] = {}
    lattice = []
    for i, j, k in itertools.product(range(n + 1), repeat=3):
        index[(i, j, k)] = len(lattice)
        lattice.append((grid[i], grid[j], grid[k]))
    reference = np.asarray(lattice, dtype=float)
    X = reference + distortion * h * _free_axes(reference) * rng.uniform(
        -1.0, 1.0, size=reference.shape
    )
    connectivity = [
        [
            index[(i, j, k)], index[(i + 1, j, k)],
            index[(i + 1, j + 1, k)], index[(i, j + 1, k)],
            index[(i, j, k + 1)], index[(i + 1, j, k + 1)],
            index[(i + 1, j + 1, k + 1)], index[(i, j + 1, k + 1)],
        ]
        for i, j, k in itertools.product(range(n), repeat=3)
    ]

    if topology == "hex20":
        points = list(X)
        references = list(reference)
        mid_index: dict[tuple[int, int], int] = {}
        extended = []
        for corners in connectivity:
            row = list(corners)
            for a in range(8, 20):
                N, _ = hex8_shape(HEX20_PARENT_NODES[a])
                ends = tuple(sorted(
                    int(corners[b]) for b in np.flatnonzero(np.abs(N) > 1.0e-12)
                ))
                if ends not in mid_index:
                    mid_reference = 0.5 * (reference[ends[0]] + reference[ends[1]])
                    # Midpoint of the two (already distorted) corner nodes,
                    # then its own masked offset: interior edges curve, while
                    # boundary mid-nodes stay inside their face plane.
                    position = 0.5 * (points[ends[0]] + points[ends[1]])
                    position = position + distortion * h * _free_axes(
                        mid_reference
                    ) * rng.uniform(-1.0, 1.0, size=3)
                    mid_index[ends] = len(points)
                    points.append(position)
                    references.append(mid_reference)
                row.append(mid_index[ends])
            extended.append(row)
        X = np.asarray(points)
        reference = np.asarray(references)
        connectivity = extended
    elif topology != "hex8":
        raise ValueError(f"unsupported patch topology {topology!r}")

    elements = {
        tag: MeshElement(tag, topology, np.asarray(conn, dtype=np.int64))
        for tag, conn in enumerate(connectivity, start=1)
    }
    node_groups = {}
    for axis, (low, high) in enumerate((("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))):
        node_groups[low] = np.flatnonzero(reference[:, axis] < 1.0e-12).astype(np.int64)
        node_groups[high] = np.flatnonzero(reference[:, axis] > 1.0 - 1.0e-12).astype(np.int64)
    node_tags = np.arange(1, len(X) + 1, dtype=np.int64)
    mesh = Mesh(
        X,
        node_tags,
        {int(tag): i for i, tag in enumerate(node_tags)},
        elements,
        {"solid": np.asarray(sorted(elements), dtype=np.int64)},
        node_groups,
        {"solid": 3, **{name: 2 for name in _FACES}},
        {},
    )
    return mesh, reference


def _interior_dof_mask(mesh: Mesh) -> np.ndarray:
    boundary = np.unique(np.concatenate([mesh.node_groups[name] for name in _FACES]))
    nodes = np.ones(len(mesh.X), dtype=bool)
    nodes[boundary] = False
    return np.repeat(nodes, 3)


def _patch_deck(
    formulation: str,
    material_block: str,
    directory: Path,
    extra: str = "",
    verification: str = "\n[verification]\n",
):
    text = (
        _DECK_TEMPLATE
        .replace("@MATERIAL@", material_block)
        .replace("@FORMULATION@", formulation)
        .replace("@OUTPUT@", str(directory / "results"))
        + extra
        + verification
    )
    deck_path = directory / f"patch_{formulation}.toml"
    deck_path.write_text(text)
    return load_deck(deck_path)


def _patch_model(formulation: str, material_block: str, mesh: Mesh, directory: Path):
    return build_model(_patch_deck(formulation, material_block, directory), mesh)


def analytic_internal_force(mesh: Mesh, topology: str, P: np.ndarray) -> np.ndarray:
    """Independent oracle for the residual under a homogeneous deformation.

    With P constant, f_A = P : int(grad_X N_A) dV0 exactly.  This uses only the
    shape functions and the material response -- never the element kernel's
    stress transform, B matrix, Voigt packing or current-volume measure -- so
    it detects defects that leave the stress merely *constant* rather than
    correct, which the interior-residual identity alone cannot see.
    """
    shape = shape_for(topology)
    points = HEX20_POINTS if topology == "hex20" else HEX8_POINTS
    weights = HEX20_WEIGHTS if topology == "hex20" else HEX8_WEIGHTS
    f = np.zeros(mesh.ndof)
    for element in mesh.elements.values():
        conn = element.connectivity
        X = mesh.X[conn]
        local = np.zeros((len(conn), 3))
        for xi, weight in zip(points, weights):
            _, dN = shape(xi)
            J0 = X.T @ dN
            grad_X = dN @ np.linalg.inv(J0)
            local += (
                np.einsum("iI,aI->ai", P, grad_X)
                * float(np.linalg.det(J0)) * float(weight)
            )
        f[(3 * conn[:, None] + np.arange(3)).ravel()] += local.ravel()
    return f


def _neo_stress(F: np.ndarray, kappa: float) -> np.ndarray:
    return update_neo_hook(
        MaterialRequest(
            np.eye(3), F, StateLayout().view(np.empty(0)),
            {"mu": 1.0, "kappa": kappa}, None, 0.0, 1.0, False,
        )
    ).P


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


class PatchMeshGuardTests(unittest.TestCase):
    """Self-guards: a patch test on an accidentally regular mesh would pass
    trivially, so the mesh itself is asserted to be distorted and valid."""

    def test_mesh_is_distorted_valid_and_has_interior_nodes(self) -> None:
        for topology, seed in itertools.product(("hex8", "hex20"), _SEEDS):
            with self.subTest(topology=topology, seed=seed):
                mesh, reference = build_patch_mesh(topology, seed)
                free = _free_axes(reference)
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

    def check_homogeneous(self, formulation, material_block, F, P, seeds=_SEEDS):
        topology = "hex20" if formulation == "hex20" else "hex8"
        for seed in seeds:
            with self.subTest(formulation=formulation, seed=seed):
                mesh, _ = build_patch_mesh(topology, seed)
                result, F_error, J_error, stresses = self.assemble_patch(
                    formulation, material_block, mesh, F
                )
                interior = _interior_dof_mask(mesh)
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
            "hex8", _NEO_BLOCK.replace("@KAPPA@", "20.0"), _F_FINITE,
            _neo_stress(_F_FINITE, 20.0),
        )

    def test_hex8_fbar_finite_deformation(self) -> None:
        # Homogeneous deformation gives alpha == 1: this certifies F-bar does
        # not break the constant state; the projection itself is verified by
        # test_fbar_spatial_and_reference_forms_agree in test_numerics.py.
        self.check_homogeneous(
            "hex8_fbar", _NEO_BLOCK.replace("@KAPPA@", "20.0"), _F_FINITE,
            _neo_stress(_F_FINITE, 20.0),
        )

    def test_hex8_fbar_near_incompressible(self) -> None:
        self.check_homogeneous(
            "hex8_fbar", _NEO_BLOCK.replace("@KAPPA@", "1.0e4"), _F_FINITE,
            _neo_stress(_F_FINITE, 1.0e4), seeds=(_SEEDS[0],),
        )

    def test_hex20_finite_deformation(self) -> None:
        self.check_homogeneous(
            "hex20", _NEO_BLOCK.replace("@KAPPA@", "20.0"), _F_FINITE,
            _neo_stress(_F_FINITE, 20.0),
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
                    "hex20" if formulation == "hex20" else "hex8", _SEEDS[0]
                )
                u = (mesh.X @ R.T + translation - mesh.X).ravel()
                with tempfile.TemporaryDirectory() as scratch:
                    model = _patch_model(
                        formulation, _NEO_BLOCK.replace("@KAPPA@", "20.0"), mesh, Path(scratch)
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
                mesh, _ = build_patch_mesh("hex8", _SEEDS[0])
                result, F_error, J_error, stresses = self.assemble_patch(
                    formulation, _J2_BLOCK, mesh, F
                )
                interior = _interior_dof_mask(mesh)
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


def _affine_constraint_block(F: np.ndarray) -> str:
    """Drive all six faces with macro_F ramped from I to F over t in [0, 1]."""
    rows = []
    for i in range(3):
        entries = []
        for j in range(3):
            constant = 1.0 if i == j else 0.0
            scale = float(F[i, j]) - constant
            entries.append(
                f'{{constant = {constant}, curve = "ramp", scale = {scale!r}}}'
            )
        rows.append("  [" + ", ".join(entries) + "]")
    matrix = ",\n".join(rows)
    faces = ", ".join(f'"{name}"' for name in _FACES)
    return (
        '\n[[curves]]\nname = "ramp"\npoints = [[0.0, 0.0], [1.0, 1.0]]\n'
        "\n[[constraints.affine]]\n"
        'name = "patch_boundary"\n'
        f"regions = [{faces}]\n"
        "origin = [0.0, 0.0, 0.0]\n"
        f"macro_F = [\n{matrix}\n]\n"
    )


class SolvedPatchTests(unittest.TestCase):
    """PT-1b: the same patch driven through the full deck -> KKT -> Newton
    path.  Newton tolerances are tightened well past deck defaults so the
    measured error reports the element's accuracy, not the stopping rule."""

    def solve_patch(self, formulation: str, F: np.ndarray, directory: Path):
        topology = "hex20" if formulation == "hex20" else "hex8"
        mesh, _ = build_patch_mesh(topology, _SEEDS[0])
        deck = _patch_deck(
            formulation, _NEO_BLOCK.replace("@KAPPA@", "20.0"), directory,
            extra=_affine_constraint_block(F), verification=_VERIFICATION_BLOCK,
        )
        events, output_times, restart_times = mandatory_events(deck)
        prepared = PreparedAnalysis(
            deck, mesh, build_model(deck, mesh), build_constraints(deck, mesh),
            events, output_times, restart_times,
        )
        return mesh, run_analysis(prepared)

    def test_solved_patch_reproduces_the_affine_field(self) -> None:
        for formulation in ("hex8", "hex8_fbar", "hex20"):
            with self.subTest(formulation=formulation):
                with tempfile.TemporaryDirectory() as scratch:
                    mesh, result = self.solve_patch(
                        formulation, _F_FINITE, Path(scratch)
                    )
                expected = ((_F_FINITE - np.eye(3)) @ mesh.X.T).T.ravel()
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
                    "hex20" if formulation == "hex20" else "hex8", _SEEDS[0]
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
