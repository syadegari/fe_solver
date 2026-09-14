"""Extract kinematic and phase-average histories from a periodic composite run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fe_solver.config import Curve, Deck
from fe_solver.constraints import macro_deformation_function
from fe_solver.io import RESULT_SCHEMA_VERSION
from fe_solver.output_fields import unpack_symmetric
from fe_solver.quadrature import HEX8_POINTS, HEX8_WEIGHTS
from fe_solver.shape import hex8_shape
from fe_solver.types import ModelError


_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _deck_from_resolved_input(database: Path, data: dict[str, Any]) -> Deck:
    curves: dict[str, Curve] = {}
    for entry in data.get("curves", []):
        points = np.asarray(entry["points"], dtype=float)
        name = str(entry["name"])
        curves[name] = Curve(name, points[:, 0], points[:, 1])
    return Deck(database, data, curves)


def _loading_component(entry: dict[str, Any]) -> tuple[int, int, str]:
    deformation = entry.get("macro_deformation", {})
    kind = str(deformation.get("type", ""))
    if kind == "isochoric_uniaxial":
        axis = str(deformation["axis"])
        index = _COMPONENT_INDEX[axis]
        return index, index, kind
    if kind == "simple_shear":
        direction = _COMPONENT_INDEX[str(deformation["direction"])]
        normal = _COMPONENT_INDEX[str(deformation["normal"])]
        return direction, normal, kind
    raise ModelError(
        "periodic composite extraction supports isochoric_uniaxial or simple_shear"
    )


def extract_periodic_composite_history(database: str | Path) -> dict[str, Any]:
    """Reconstruct average raw F and homogenized response from accepted states."""
    database_path = Path(database).resolve()
    with h5py.File(database_path, "r") as archive:
        version = int(archive.attrs.get("schema_version", -1))
        if version != RESULT_SCHEMA_VERSION:
            raise ModelError(
                f"results database schema version {version} is unsupported; "
                f"version {RESULT_SCHEMA_VERSION} is required"
            )
        complete = int(archive["results"].attrs.get("n_complete_steps", -1))
        if complete < 1:
            raise ModelError("results database contains no complete accepted state")
        times = np.asarray(archive["results/time"][:complete], dtype=float)
        X = np.asarray(archive["mesh/reference_coordinates"], dtype=float)
        displacement = np.asarray(
            archive["results/nodal/displacement"][:complete], dtype=float
        )
        reactions = np.asarray(
            archive["results/nodal/constraint_reaction"][:complete], dtype=float
        )
        resolved = json.loads(_decode(archive["meta/resolved_input_json"][()]))
        analysis_name = str(archive["meta"].attrs["analysis_name"])

        block_data = []
        state_fields: dict[str, list[str]] = {}
        formulations: dict[str, str] = {}
        for name in sorted(archive["mesh/blocks"]):
            mesh_block = archive[f"mesh/blocks/{name}"]
            result_block = archive[f"results/blocks/{name}"]
            formulation = str(mesh_block.attrs["formulation"])
            if formulation not in {"hex8", "hex8_fbar"}:
                raise ModelError(
                    "periodic composite extraction requires eight-node hex8 or "
                    "hex8_fbar blocks"
                )
            region = str(mesh_block.attrs["region"])
            connectivity = np.asarray(mesh_block["connectivity"], dtype=np.int64)
            if connectivity.ndim != 2 or connectivity.shape[1] != 8:
                raise ModelError(
                    "periodic composite extraction requires eight-node connectivity"
                )
            stress = np.asarray(result_block["cauchy_stress"][:complete], dtype=float)
            fields = sorted(result_block["state"].keys()) if "state" in result_block else []
            state_fields[region] = fields
            formulations[region] = formulation
            block_data.append((region, connectivity, stress))

    periodic = resolved.get("constraints", {}).get("periodic_rve", [])
    if len(periodic) != 1:
        raise ModelError("periodic composite extraction requires one periodic_rve entry")
    entry = periodic[0]
    deck = _deck_from_resolved_input(database_path, resolved)
    macro = macro_deformation_function(entry, deck)
    component_i, component_j, loading_kind = _loading_component(entry)

    derivatives = np.asarray([hex8_shape(point)[1] for point in HEX8_POINTS])
    weights = np.asarray(HEX8_WEIGHTS, dtype=float)
    prepared_blocks = []
    reference_volume = 0.0
    for region, connectivity, stress in block_data:
        X_e = X[connectivity]
        J0 = np.einsum("eni,gnj->egij", X_e, derivatives)
        detJ0 = np.linalg.det(J0)
        if np.any(detJ0 <= 0.0):
            raise ModelError("results database contains a nonpositive reference mapping")
        dv0 = detJ0 * weights[None, :]
        reference_volume += float(np.sum(dv0))
        prepared_blocks.append(
            (region, connectivity, stress, X_e, np.linalg.inv(J0), dv0)
        )

    average_F = np.empty((complete, 3, 3))
    phase_stress = {
        region: np.empty((complete, 3, 3)) for region, *_ in prepared_blocks
    }
    total_stress = np.empty((complete, 3, 3))
    for step in range(complete):
        integrated_F = np.zeros((3, 3))
        integrated_stress = np.zeros((3, 3))
        current_volume = 0.0
        for region, connectivity, stress, X_e, invJ0, dv0 in prepared_blocks:
            x_e = X_e + displacement[step, connectivity]
            Jx = np.einsum("eni,gnj->egij", x_e, derivatives)
            F = np.einsum("egij,egjk->egik", Jx, invJ0)
            integrated_F += np.einsum("egij,eg->ij", F, dv0)
            element_volume = np.sum(np.linalg.det(F) * dv0, axis=1)
            sigma = unpack_symmetric(stress[step])
            phase_volume = float(np.sum(element_volume))
            phase_integral = np.einsum("e,eij->ij", element_volume, sigma)
            phase_stress[region][step] = phase_integral / phase_volume
            integrated_stress += phase_integral
            current_volume += phase_volume
        average_F[step] = integrated_F / reference_volume
        total_stress[step] = integrated_stress / current_volume

    macro_F = np.asarray([macro(float(time)) for time in times])
    average_F_error = np.max(np.abs(average_F - macro_F), axis=(1, 2))
    # Stored constraint reactions equal the assembled internal nodal forces at
    # equilibrium, so their reference-coordinate moment gives average P.
    macro_P = np.einsum("tni,nj->tij", reactions, X) / reference_volume
    deformation_component = macro_F[:, component_i, component_j]

    return {
        "database": str(database_path),
        "analysis_name": analysis_name,
        "loading_kind": loading_kind,
        "component": [component_i + 1, component_j + 1],
        "time": times.tolist(),
        "macro_deformation_component": deformation_component.tolist(),
        "macro_nominal_stress_component": macro_P[:, component_i, component_j].tolist(),
        "phase_cauchy_stress_component": {
            region: values[:, component_i, component_j].tolist()
            for region, values in phase_stress.items()
        },
        "whole_cell_cauchy_stress_component": total_stress[
            :, component_i, component_j
        ].tolist(),
        "volume_average_F": average_F.tolist(),
        "prescribed_macro_F": macro_F.tolist(),
        "volume_average_F_error_inf": average_F_error.tolist(),
        "maximum_volume_average_F_error_inf": float(np.max(average_F_error)),
        "state_fields_by_region": state_fields,
        "formulations_by_region": formulations,
    }


def plot_periodic_composite_history(
    history: dict[str, Any], output: str | Path
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deformation = np.asarray(history["macro_deformation_component"], dtype=float)
    component = "".join(str(value) for value in history["component"])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(
        deformation,
        history["macro_nominal_stress_component"],
        label=rf"$\overline{{P}}_{{{component}}}$",
    )
    axes[0].set(
        xlabel=rf"prescribed $\overline{{F}}_{{{component}}}$",
        ylabel="macroscopic nominal stress [MPa]",
    )
    for region, values in history["phase_cauchy_stress_component"].items():
        axes[1].plot(deformation, values, label=region)
    axes[1].plot(
        deformation,
        history["whole_cell_cauchy_stress_component"],
        color="black",
        linestyle="--",
        label="whole cell",
    )
    axes[1].set(
        xlabel=rf"prescribed $\overline{{F}}_{{{component}}}$",
        ylabel=rf"phase-average $\sigma_{{{component}}}$ [MPa]",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle(str(history["analysis_name"]))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="check a periodic mixed-material result database"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    history = extract_periodic_composite_history(args.database)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        plot_periodic_composite_history(history, args.plot)
    print(
        f"{history['analysis_name']}: "
        f"max|<F_raw>-Fbar|={history['maximum_volume_average_F_error_inf']:.3e}; "
        f"final macro P={history['macro_nominal_stress_component'][-1]:.6g} MPa"
    )


if __name__ == "__main__":
    main()
