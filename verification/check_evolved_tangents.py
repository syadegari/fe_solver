"""Check material and F-bar tangents using a committed restart state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from fe_solver.assembly import element_dofs
from fe_solver.elements import _kinematics, evaluate_element
from fe_solver.io import load_restart
from fe_solver.materials import evaluate_material_point
from fe_solver.preprocess import prepare_analysis
from fe_solver.quadrature import HEX8_POINTS
from fe_solver.shape import hex8_shape
from fe_solver.types import ElementRequest, MaterialRequest, ModelError


def _material_kinematics(
    X: np.ndarray, u_n: np.ndarray, u_trial: np.ndarray, formulation: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    x_n = X + u_n.reshape(-1, 3)
    x_trial = X + u_trial.reshape(-1, 3)
    if formulation == "hex8_fbar":
        _, dN_c = hex8_shape(np.zeros(3))
        *_, Fc_n = _kinematics(X, x_n, dN_c)
        *_, Fc = _kinematics(X, x_trial, dN_c)
        Jc_n = float(np.linalg.det(Fc_n))
        Jc = float(np.linalg.det(Fc))
    values = []
    for point in HEX8_POINTS:
        _, dN = hex8_shape(point)
        *_, Fg_n = _kinematics(X, x_n, dN)
        *_, Fg = _kinematics(X, x_trial, dN)
        if formulation == "hex8_fbar":
            Fg_n = (Jc_n / np.linalg.det(Fg_n)) ** (1.0 / 3.0) * Fg_n
            Fg = (Jc / np.linalg.det(Fg)) ** (1.0 / 3.0) * Fg
        values.append((Fg_n, Fg))
    return values


def _relative_inf(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(error, ord=np.inf) / max(np.linalg.norm(reference, ord=np.inf), 1.0))


def check_material_tangent(
    material, state_e_n: np.ndarray, kinematics: list[tuple[np.ndarray, np.ndarray]],
    t_n: float, t_trial: float, rng: np.random.Generator, directions: int,
) -> dict:
    errors = []
    plastic_strains = []
    eps = 2.0e-7
    for g, (F_n, F) in enumerate(kinematics):
        state_n = material.state_layout.view(state_e_n[g])
        request = MaterialRequest(
            F_n, F, state_n, material.properties, None, t_n, t_trial, True
        )
        response = evaluate_material_point(material, request)
        if not response.status.ok:
            raise ModelError(response.status.message)
        plastic_strains.append(float(response.state_trial["equivalent_plastic_strain"]))
        assert response.A_alg is not None
        for _ in range(directions):
            direction = rng.normal(size=(3, 3))
            direction /= np.linalg.norm(direction)

            def stress(F_value: np.ndarray) -> np.ndarray:
                result = evaluate_material_point(
                    material,
                    MaterialRequest(
                        F_n, F_value, state_n, material.properties, None,
                        t_n, t_trial, False,
                    ),
                )
                if not result.status.ok:
                    raise ModelError(result.status.message)
                return result.P

            finite_difference = (stress(F + eps * direction) - stress(F - eps * direction)) / (2.0 * eps)
            analytic = np.einsum("iIjJ,jJ->iI", response.A_alg, direction)
            errors.append(_relative_inf(analytic - finite_difference, finite_difference))
    return {
        "directions": len(errors),
        "maximum_relative_inf_error": max(errors),
        "median_relative_inf_error": float(np.median(errors)),
        "trial_equivalent_plastic_strain_range": [min(plastic_strains), max(plastic_strains)],
    }


def check_element_tangent(
    request: ElementRequest, rng: np.random.Generator, directions: int,
) -> dict:
    response = evaluate_element(request)
    if not response.status.ok:
        raise ModelError(response.status.message)
    assert response.K is not None
    errors = []
    step_sizes = [3.0e-6, 1.0e-6, 3.0e-7]
    for _ in range(directions):
        direction = rng.normal(size=request.u_e_trial.size)
        direction /= np.linalg.norm(direction)
        analytic = response.K @ direction
        candidates = []
        for eps in step_sizes:
            def force(displacement: np.ndarray) -> np.ndarray:
                trial = evaluate_element(
                    ElementRequest(
                        request.X_e, request.u_e_n, displacement, request.state_e_n,
                        request.material, request.point_properties, request.t_n,
                        request.t_np1, False, request.formulation,
                    )
                )
                if not trial.status.ok:
                    raise ModelError(trial.status.message)
                return trial.f_int

            finite_difference = (
                force(request.u_e_trial + eps * direction)
                - force(request.u_e_trial - eps * direction)
            ) / (2.0 * eps)
            candidates.append(_relative_inf(analytic - finite_difference, finite_difference))
        errors.append(min(candidates))
    return {
        "directions": len(errors),
        "maximum_relative_inf_error": max(errors),
        "median_relative_inf_error": float(np.median(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path)
    parser.add_argument("restart", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--trial-time", type=float, default=0.405)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepared = prepare_analysis(args.deck)
    t_n, _, u_n, _ = load_restart(args.restart, prepared.model)
    with h5py.File(args.database, "r") as archive:
        complete = int(archive["results"].attrs["n_complete_steps"])
        times = np.asarray(archive["results/time"][:complete])
        index = int(np.argmin(np.abs(times - args.trial_time)))
        if abs(times[index] - args.trial_time) > 1.0e-10:
            raise ModelError(f"database has no accepted state at t={args.trial_time}")
        u_trial = np.asarray(archive["results/nodal/displacement"][index]).ravel()
        ep = np.asarray(
            archive["results/blocks/0000/state/equivalent_plastic_strain"][index]
        )
    block = prepared.model.blocks[0]
    centers = prepared.mesh.X[block.connectivity].mean(axis=1)
    neck_layer = np.flatnonzero(
        np.isclose(centers[:, 2], centers[:, 2].min(), rtol=0.0, atol=1.0e-10)
    )
    neck_radii = np.linalg.norm(centers[neck_layer, :2], axis=1)
    selected = {
        "maximum_plastic_strain": int(np.argmax(ep)),
        "neck_outer": int(neck_layer[np.argmax(neck_radii)]),
        "far_end": int(np.argmax(centers[:, 2])),
    }
    rng = np.random.default_rng(7201)
    report = {
        "restart_time": t_n,
        "trial_time": float(times[index]),
        "elements": {},
    }
    for label, element_index in selected.items():
        connectivity = block.connectivity[element_index]
        dofs = element_dofs(connectivity)
        request = ElementRequest(
            prepared.mesh.X[connectivity], u_n[dofs], u_trial[dofs],
            block.state_n[element_index], block.material, None, t_n,
            float(times[index]), True, block.formulation,
        )
        kinematics = _material_kinematics(
            request.X_e, request.u_e_n, request.u_e_trial, request.formulation
        )
        report["elements"][label] = {
            "element_index": element_index,
            "reference_center": centers[element_index].tolist(),
            "committed_equivalent_plastic_strain_range": [
                float(block.state_n[element_index, :, 6].min()),
                float(block.state_n[element_index, :, 6].max()),
            ],
            "material": check_material_tangent(
                block.material, block.state_n[element_index], kinematics,
                t_n, float(times[index]), rng, args.directions,
            ),
            "element": check_element_tangent(request, rng, args.directions),
        }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
