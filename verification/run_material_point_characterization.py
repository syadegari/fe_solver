"""Generate reviewable prescribed-F characterization curves for the J2 model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fe_solver.material_point import MaterialPointHistory, run_material_path
from fe_solver.materials import j2_plasticity_definition
from fe_solver.output_fields import TENSOR_COMPONENTS, pack_symmetric


PROPERTIES = {
    "shear_modulus": 80193.8,
    "bulk_modulus": 164210.0,
    "initial_yield_stress": 450.0,
    "linear_hardening_modulus": 129.24,
    "saturation_increment": 265.0,
    "saturation_rate": 16.93,
}


def _hardening(ep: np.ndarray) -> np.ndarray:
    return (
        PROPERTIES["initial_yield_stress"]
        + PROPERTIES["linear_hardening_modulus"] * ep
        + PROPERTIES["saturation_increment"]
        * (1.0 - np.exp(-PROPERTIES["saturation_rate"] * ep))
    )


def _yield_residual(history: MaterialPointHistory) -> np.ndarray:
    tau = history.kirchhoff_stress
    dev = tau - np.trace(tau, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    return np.linalg.norm(dev, axis=(1, 2)) - np.sqrt(2.0 / 3.0) * _hardening(history.state[:, 6])


def _write_csv(
    path: Path, history: MaterialPointHistory, loading: np.ndarray,
    path_conjugate_stress: np.ndarray,
) -> None:
    packed = {
        "cauchy": pack_symmetric(history.cauchy_stress),
        "kirchhoff": pack_symmetric(history.kirchhoff_stress),
        "green_lagrange": pack_symmetric(history.green_lagrange_strain),
        "euler_almansi": pack_symmetric(history.euler_almansi_strain),
        "material_log": pack_symmetric(history.material_log_strain),
        "spatial_log": pack_symmetric(history.spatial_log_strain),
    }
    fieldnames = ["step", "time", "loading", "path_conjugate_stress"]
    for name in packed:
        fieldnames.extend(f"{name}_{component}" for component in TENSOR_COMPONENTS)
    fieldnames.extend(
        [
            "equivalent_plastic_strain", "yield_residual",
            "A_1111", "A_1122", "A_1212", "D_1111", "D_1122", "D_1212",
        ]
    )
    residual = _yield_residual(history)
    assert history.A_alg is not None and history.spatial_truesdell_tangent is not None
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(len(history.time)):
            row = {
                "step": index,
                "time": history.time[index],
                "loading": loading[index],
                "path_conjugate_stress": path_conjugate_stress[index],
                "equivalent_plastic_strain": history.state[index, 6],
                "yield_residual": residual[index],
                "A_1111": history.A_alg[index, 0, 0, 0, 0],
                "A_1122": history.A_alg[index, 0, 0, 1, 1],
                "A_1212": history.A_alg[index, 0, 1, 0, 1],
                "D_1111": history.spatial_truesdell_tangent[index, 0, 0],
                "D_1122": history.spatial_truesdell_tangent[index, 0, 1],
                "D_1212": history.spatial_truesdell_tangent[index, 3, 3],
            }
            for name, values in packed.items():
                row.update(
                    {f"{name}_{component}": values[index, j] for j, component in enumerate(TENSOR_COMPONENTS)}
                )
            writer.writerow(row)


def _save_full(path: Path, history: MaterialPointHistory) -> None:
    np.savez_compressed(
        path,
        path_parameter=history.path_parameter,
        time=history.time,
        F=history.F,
        P=history.P,
        kirchhoff_stress=history.kirchhoff_stress,
        cauchy_stress=history.cauchy_stress,
        green_lagrange_strain=history.green_lagrange_strain,
        euler_almansi_strain=history.euler_almansi_strain,
        material_log_strain=history.material_log_strain,
        spatial_log_strain=history.spatial_log_strain,
        state=history.state,
        A_alg=history.A_alg,
        spatial_truesdell_tangent=history.spatial_truesdell_tangent,
    )


def _first_plastic(history: MaterialPointHistory) -> int:
    indices = np.flatnonzero(history.state[:, 6] > 0.0)
    return int(indices[0]) if len(indices) else -1


def _tangent_summary(history: MaterialPointHistory) -> dict[str, dict[str, float]]:
    assert history.spatial_truesdell_tangent is not None
    components = {
        "D_1111": history.spatial_truesdell_tangent[:, 0, 0],
        "D_1122": history.spatial_truesdell_tangent[:, 0, 1],
        "D_1212": history.spatial_truesdell_tangent[:, 3, 3],
    }
    return {
        name: {
            "initial": float(values[0]),
            "final": float(values[-1]),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
        for name, values in components.items()
    }


def _plot_uniaxial(path: Path, history: MaterialPointHistory, strain: np.ndarray) -> None:
    sigma = history.cauchy_stress
    conjugate = sigma[:, 0, 0] - 0.5 * (sigma[:, 1, 1] + sigma[:, 2, 2])
    D = history.spatial_truesdell_tangent
    assert D is not None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot(strain, sigma[:, 0, 0], label=r"$\sigma_{11}$")
    axes[0].plot(strain, conjugate, "--", label="path-conjugate stress")
    axes[0].set(xlabel="axial logarithmic strain", ylabel="stress [MPa]")
    axes[0].legend()
    axes[1].plot(strain, history.state[:, 6])
    axes[1].set(xlabel="axial logarithmic strain", ylabel="equivalent plastic strain")
    axes[2].plot(strain, D[:, 0, 0], label=r"$D_{1111}$")
    axes[2].plot(strain, D[:, 0, 1], label=r"$D_{1122}$")
    axes[2].plot(strain, D[:, 3, 3], label=r"$D_{1212}$")
    axes[2].set(xlabel="axial logarithmic strain", ylabel="spatial tangent [MPa]")
    axes[2].legend()
    fig.suptitle("J2 material point: isochoric uniaxial extension")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_shear(path: Path, history: MaterialPointHistory, gamma: np.ndarray) -> None:
    e12 = history.euler_almansi_strain[:, 0, 1]
    D = history.spatial_truesdell_tangent
    assert D is not None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot(e12, history.cauchy_stress[:, 0, 1])
    axes[0].set(xlabel=r"tensorial Euler--Almansi $e_{12}$", ylabel=r"$\sigma_{12}$ [MPa]")
    axes[1].plot(gamma, history.state[:, 6])
    axes[1].set(xlabel=r"simple-shear amount $\gamma=F_{12}$", ylabel="equivalent plastic strain")
    axes[2].plot(gamma, D[:, 0, 0], label=r"$D_{1111}$")
    axes[2].plot(gamma, D[:, 0, 1], label=r"$D_{1122}$")
    axes[2].plot(gamma, D[:, 3, 3], label=r"$D_{1212}$")
    axes[2].set(xlabel=r"simple-shear amount $\gamma$", ylabel="spatial tangent [MPa]")
    axes[2].legend()
    fig.suptitle("J2 material point: simple shear")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--comparison-steps", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "material_point_results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    material = j2_plasticity_definition("voce_steel", PROPERTIES)

    def uniaxial(parameter: float) -> np.ndarray:
        strain = 0.1 * parameter
        return np.diag([np.exp(strain), np.exp(-0.5 * strain), np.exp(-0.5 * strain)])

    def shear(parameter: float) -> np.ndarray:
        F = np.eye(3)
        F[0, 1] = 0.1 * parameter
        return F

    uniaxial_history = run_material_path(material, uniaxial, args.steps)
    shear_history = run_material_path(material, shear, args.steps)
    uniaxial_coarse = run_material_path(
        material, uniaxial, args.comparison_steps, need_tangent=False
    )
    shear_coarse = run_material_path(
        material, shear, args.comparison_steps, need_tangent=False
    )
    strain = 0.1 * uniaxial_history.path_parameter
    gamma = 0.1 * shear_history.path_parameter
    uniaxial_conjugate = (
        uniaxial_history.cauchy_stress[:, 0, 0]
        - 0.5 * (
            uniaxial_history.cauchy_stress[:, 1, 1]
            + uniaxial_history.cauchy_stress[:, 2, 2]
        )
    )
    shear_conjugate = shear_history.P[:, 0, 1]
    _write_csv(args.output / "j2_uniaxial.csv", uniaxial_history, strain, uniaxial_conjugate)
    _write_csv(args.output / "j2_simple_shear.csv", shear_history, gamma, shear_conjugate)
    _save_full(args.output / "j2_uniaxial.npz", uniaxial_history)
    _save_full(args.output / "j2_simple_shear.npz", shear_history)
    _plot_uniaxial(args.output / "j2_uniaxial.png", uniaxial_history, strain)
    _plot_shear(args.output / "j2_simple_shear.png", shear_history, gamma)

    mu = PROPERTIES["shear_modulus"]
    bulk = PROPERTIES["bulk_modulus"]
    summary = {
        "properties": PROPERTIES,
        "steps": args.steps,
        "comparison_steps": args.comparison_steps,
        "elastic_reference_MPa": {
            "D_1111": bulk + 4.0 * mu / 3.0,
            "D_1122": bulk - 2.0 * mu / 3.0,
            "D_1212": mu,
        },
        "uniaxial": {
            "first_plastic_step": _first_plastic(uniaxial_history),
            "final_sigma_11_MPa": float(uniaxial_history.cauchy_stress[-1, 0, 0]),
            "final_equivalent_plastic_strain": float(uniaxial_history.state[-1, 6]),
            "step_refinement_stress_relative": float(
                np.linalg.norm(uniaxial_history.cauchy_stress[-1] - uniaxial_coarse.cauchy_stress[-1])
                / np.linalg.norm(uniaxial_history.cauchy_stress[-1])
            ),
            "step_refinement_state_relative": float(
                np.linalg.norm(uniaxial_history.state[-1] - uniaxial_coarse.state[-1])
                / np.linalg.norm(uniaxial_history.state[-1])
            ),
            "spatial_tangent_components_MPa": _tangent_summary(uniaxial_history),
        },
        "simple_shear": {
            "first_plastic_step": _first_plastic(shear_history),
            "final_sigma_12_MPa": float(shear_history.cauchy_stress[-1, 0, 1]),
            "final_equivalent_plastic_strain": float(shear_history.state[-1, 6]),
            "step_refinement_stress_relative": float(
                np.linalg.norm(shear_history.cauchy_stress[-1] - shear_coarse.cauchy_stress[-1])
                / np.linalg.norm(shear_history.cauchy_stress[-1])
            ),
            "step_refinement_state_relative": float(
                np.linalg.norm(shear_history.state[-1] - shear_coarse.state[-1])
                / np.linalg.norm(shear_history.state[-1])
            ),
            "spatial_tangent_components_MPa": _tangent_summary(shear_history),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
