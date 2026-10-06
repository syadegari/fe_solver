"""Plot accepted TRIP RVE histories without loading the constitutive library.

Cauchy stress uses current-volume weights; scalar state uses reference-volume
weights. Both averages use the saved centroid-recovered fields, not a fresh
constitutive evaluation or an exact integration of Gauss-point histories.
The supplied TRIP constants use GPa; plotted/exported stresses are MPa.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

from fe_solver.quadrature import HEX8_POINTS, HEX8_WEIGHTS
from fe_solver.output_fields import TENSOR_COMPONENTS, pack_symmetric
from fe_solver.shape import hex8_shape
from fe_solver.types import ModelError
from verification.check_periodic_composite import extract_periodic_composite_history, von_mises_stress


def extract_trip_history(
    database: str | Path, *, martensite_field: str = "martensite_fraction"
) -> dict[str, Any]:
    """Return stress and scalar-state curves, with explicit averaging measures."""
    database = Path(database)
    response = extract_periodic_composite_history(database)
    times = np.asarray(response["time"], dtype=float)
    count = len(times)
    shear = response["loading_kind"] == "simple_shear"
    strain = np.asarray(response["macro_deformation_component"], dtype=float)
    if not shear:
        strain = strain - 1.0
    strain_name = "engineering_shear_gamma" if shear else "engineering_axial_strain"
    component = "".join(str(value) for value in response["component"])
    macro_F = np.asarray(response["prescribed_macro_F"], dtype=float)
    average_F = np.asarray(response["volume_average_F"], dtype=float)
    # Spatial Hencky strain of the prescribed macro deformation, not <h_local>.
    B = macro_F @ np.swapaxes(macro_F, 1, 2)
    stretches_squared, directions = np.linalg.eigh(B)
    if np.any(stretches_squared <= 0.0):
        raise ModelError("macro deformation has nonpositive squared stretches")
    h = 0.5 * np.einsum(
        "tik,tk,tjk->tij", directions, np.log(stretches_squared), directions
    )
    component_i, component_j = (value - 1 for value in response["component"])
    log_strain_name = f"macro_hencky_{component}"
    deviatoric_h = h - np.trace(h, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    columns = {
        "time_s": times,
        strain_name: strain,
        log_strain_name: h[:, component_i, component_j],
        "macro_equivalent_hencky_strain": np.sqrt(
            (2.0 / 3.0) * np.sum(deviatoric_h**2, axis=(1, 2))
        ),
        "volume_average_F_error_inf": np.asarray(response["volume_average_F_error_inf"]),
        f"macro_nominal_stress_{component}_MPa": 1000.0 * np.asarray(
            response["macro_nominal_stress_component"]
        ),
    }
    # Plot shear in engineering-like Hencky scaling; retain the tensor component.
    plotted_strain_name = f"twice_{log_strain_name}" if shear else log_strain_name
    if shear:
        columns[plotted_strain_name] = 2.0 * columns[log_strain_name]
    stress_tensors = {
        "rve": np.asarray(response["whole_cell_cauchy_stress_tensor"]),
        **{
            region: np.asarray(stress)
            for region, stress in response["phase_cauchy_stress_tensor"].items()
        },
    }
    for region, stress in stress_tensors.items():
        packed = 1000.0 * pack_symmetric(stress)
        for index, label in enumerate(TENSOR_COMPONENTS):
            columns[f"{region}_cauchy_stress_{label}_MPa"] = packed[:, index]
        # Trace is linear: hydrostatic stress commutes with volume averaging.
        columns[f"{region}_hydrostatic_stress_MPa"] = (
            1000.0 * np.trace(stress, axis1=-2, axis2=-1) / 3.0
        )
        # These two operations do not commute for heterogeneous stress fields.
        columns[f"{region}_von_mises_of_mean_stress_MPa"] = 1000.0 * von_mises_stress(stress)
        local_mean = (
            response["whole_cell_mean_local_von_mises_stress"] if region == "rve"
            else response["phase_mean_local_von_mises_stress"][region]
        )
        columns[f"{region}_mean_local_von_mises_stress_MPa"] = 1000.0 * np.asarray(local_mean)
    for i in range(3):
        for j in range(3):
            columns[f"average_raw_F_{i + 1}{j + 1}"] = average_F[:, i, j]
            columns[f"prescribed_macro_F_{i + 1}{j + 1}"] = macro_F[:, i, j]

    derivatives = np.asarray([hex8_shape(point)[1] for point in HEX8_POINTS])
    volumes: dict[str, float] = {}
    beta_columns: dict[str, str] = {}
    transformed_integral = np.zeros(count)
    parent_volume = 0.0
    with h5py.File(database, "r") as archive:
        X = np.asarray(archive["mesh/reference_coordinates"], dtype=float)
        source_commit = str(archive.attrs.get("git_commit", "unknown"))
        resolved = json.loads(archive["meta/resolved_input_json"][()])
        for key, block in archive["mesh/blocks"].items():
            region = str(block.attrs["region"])
            if region in volumes:
                raise ModelError("TRIP report requires unique region names per block")
            connectivity = np.asarray(block["connectivity"], dtype=np.int64)
            J0 = np.einsum("eni,gnj->egij", X[connectivity], derivatives)
            detJ0 = np.linalg.det(J0)
            if np.any(detJ0 <= 0.0):
                raise ModelError("nonpositive reference element mapping")
            element_volume = np.sum(detJ0 * np.asarray(HEX8_WEIGHTS)[None, :], axis=1)
            phase_volume = float(np.sum(element_volume))
            volumes[region] = phase_volume
            state = archive[f"results/blocks/{key}/state"]
            if martensite_field in state and state[martensite_field].ndim != 2:
                raise ModelError("martensite output must be a scalar cell field")
            for name, dataset in state.items():
                if dataset.ndim != 2:
                    continue  # This report averages scalar outputs only.
                values = np.asarray(dataset[:count], dtype=float)
                if values.shape != (count, len(element_volume)) or not np.all(np.isfinite(values)):
                    raise ModelError(f"invalid saved scalar state field {region}/{name}")
                integral = values @ element_volume
                column = f"{region}_{name}_mean_reference"
                columns[column] = integral / phase_volume
                if name == "beta" or name.startswith("beta_"):
                    if region in beta_columns:
                        raise ModelError(f"multiple beta outputs in region {region!r}")
                    beta_columns[region] = column
                if name == martensite_field:
                    transformed_integral += integral
                    parent_volume += phase_volume

    if parent_volume <= 0.0:
        raise ModelError(f"no scalar state output named {martensite_field!r} was saved")
    reference_volume = sum(volumes.values())
    columns["martensite_mean_initial_austenite_reference"] = transformed_integral / parent_volume
    columns["martensite_contribution_initial_rve_reference"] = transformed_integral / reference_volume
    for name, values in columns.items():
        if not np.all(np.isfinite(values)):
            raise ModelError(f"non-finite report column {name!r}")

    return {
        "analysis_name": response["analysis_name"],
        "display_name": re.sub("8a56f", "8A56F", response["analysis_name"], flags=re.IGNORECASE),
        "database_filename": database.name,
        "source_git_commit": source_commit,
        "loading_kind": response["loading_kind"],
        "stress_component": component,
        "stress_components": list(TENSOR_COMPONENTS),
        "stress_invariant": "von Mises = sqrt(3 J2); J2 = (s:s)/2, s = dev(sigma)",
        "hydrostatic_stress_definition": "sigma_h = tr(sigma)/3; tension positive; pressure = -sigma_h",
        "strain_column": plotted_strain_name,
        "hencky_component_column": log_strain_name,
        "engineering_strain_column": strain_name,
        "strain_measure": "spatial Hencky tensor of prescribed macro F; tensorial shear",
        "beta_columns": beta_columns,
        "stress_input_unit": "GPa",
        "stress_output_unit": "MPa",
        "stress_average_measure": "current volume, reconstructed from raw nodal kinematics",
        "state_average_measure": "reference volume",
        "field_recovery": "saved centroid values; no constitutive re-evaluation",
        "martensite_field": martensite_field,
        "reference_phase_volume_fractions": {
            name: volume / reference_volume for name, volume in volumes.items()
        },
        "initial_austenite_reference_volume_fraction": parent_volume / reference_volume,
        "maximum_volume_average_F_error_inf": response["maximum_volume_average_F_error_inf"],
        "volume_average_F_error_by_component": np.max(np.abs(average_F - macro_F), axis=0).tolist(),
        "requested_end_time_s": float(resolved["analysis"]["t_end"]),
        "saved_end_time_s": float(times[-1]),
        "columns": {name: values.tolist() for name, values in columns.items()},
    }


def write_trip_report(history: dict[str, Any], output: str | Path) -> Path:
    """Save portable numeric histories and stress, transformation, beta plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    columns = {name: np.asarray(values) for name, values in history["columns"].items()}
    np.savez_compressed(output / "history.npz", **columns)
    x = columns[history["strain_column"]]
    component = history["stress_component"]
    title = history["display_name"]
    xlabel = (
        rf"Logarithmic strain $2\bar{{h}}_{{{component}}}$"
        if history["loading_kind"] == "simple_shear"
        else rf"Logarithmic strain $\bar{{h}}_{{{component}}}=\ln\bar{{\lambda}}$"
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for region in history["reference_phase_volume_fractions"]:
        axis.plot(x, columns[f"{region}_cauchy_stress_{component}_MPa"], label=region)
    axis.plot(x, columns[f"rve_cauchy_stress_{component}_MPa"], color="black", label="whole RVE", linewidth=2)
    axis.set(xlabel=xlabel, ylabel=rf"Current-volume mean Cauchy stress $\sigma_{{{component}}}$ [MPa]")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(output / "stress_vs_strain.svg")
    plt.close(figure)

    stress_regions = [*history["reference_phase_volume_fractions"], "rve"]
    styles = {"rve": {"color": "black", "linewidth": 2}}
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), sharex=True, constrained_layout=True)
    for axis, label in zip(axes.flat, history["stress_components"]):
        for region in stress_regions:
            axis.plot(
                x, columns[f"{region}_cauchy_stress_{label}_MPa"],
                label="whole RVE" if region == "rve" else region,
                **styles.get(region, {}),
            )
        axis.set_title(rf"$\sigma_{{{label}}}$")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle(title)
    figure.supxlabel(xlabel)
    figure.supylabel("Current-volume mean Cauchy stress [MPa]")
    figure.savefig(output / "stress_components_vs_strain.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, constrained_layout=True)
    for axis, suffix, heading in zip(
        axes,
        ("von_mises_of_mean_stress_MPa", "mean_local_von_mises_stress_MPa"),
        ("Von Mises of mean stress", "Mean of local von Mises stress"),
    ):
        for region in stress_regions:
            axis.plot(
                x, columns[f"{region}_{suffix}"],
                label="whole RVE" if region == "rve" else region,
                **styles.get(region, {}),
            )
        axis.set_title(heading)
        axis.set_ylabel("Equivalent stress [MPa]")
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle(title)
    figure.supxlabel(xlabel)
    figure.savefig(output / "von_mises_vs_strain.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for region in stress_regions:
        axis.plot(
            x, columns[f"{region}_hydrostatic_stress_MPa"],
            label="whole RVE" if region == "rve" else region,
            **styles.get(region, {}),
        )
    axis.set(
        xlabel=xlabel,
        ylabel=r"Current-volume mean hydrostatic stress $\sigma_{\mathrm{h}}$ [MPa]",
    )
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(output / "hydrostatic_stress_vs_strain.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(x, 100 * columns["martensite_mean_initial_austenite_reference"], label="mean within initial austenite")
    axis.plot(x, 100 * columns["martensite_contribution_initial_rve_reference"], label="contribution per initial RVE volume")
    axis.set(xlabel=xlabel, ylabel=r"Reference-volume mean stored $\Sigma\xi$ [%]")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(output / "martensite_vs_strain.svg")
    plt.close(figure)

    if history["beta_columns"]:
        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for region, name in history["beta_columns"].items():
            axis.plot(x, columns[name], label=region)
        axis.set(xlabel=xlabel, ylabel=r"Reference-volume mean $\beta$")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.savefig(output / "beta_vs_strain.svg")
        plt.close(figure)

    fractions = history["reference_phase_volume_fractions"]
    fractions_text = ", ".join(f"{name}: {100 * value:.6g}%" for name, value in fractions.items())
    readme = f"""# {title}

Saved interval: {columns['time_s'][0]:g} to {history['saved_end_time_s']:g} seconds;
requested end: {history['requested_end_time_s']:g} seconds. There are {len(x)} saved states.
Source run commit: `{history['source_git_commit']}`.
Initial phase volume fractions: {fractions_text}.

Plots are saved in SVG format only.

`stress_vs_strain.svg`: Cauchy stress component {component}, averaged
over each phase and the entire RVE using current element volumes reconstructed
from the saved nodal displacement. Native TRIP stress is GPa; report stress is MPa.
The horizontal axis is hbar11 = ln(lambdabar) for isochoric tension, and 2 hbar12
for simple shear. Here hbar = (1/2) log(Fbar Fbar^T), so
hbar12 = asinh(gamma/2) / sqrt(1 + gamma^2/4). The factor of two is only the shear
plot-axis convention; the original hbar12 component is also retained in the data.
It is not log(1+gamma), nor an average of local logarithmic strains.
These scalar plots are not presented as work-conjugate stress/strain pairs.
The original engineering strain and equivalent macro Hencky strain remain in
the saved numerical data.

To evaluate the matrix logarithm, diagonalize Bbar = Fbar Fbar^T as
Bbar = Q diag(b1,b2,b3) Q^T, then form
h = Q diag(0.5 ln(b1), 0.5 ln(b2), 0.5 ln(b3)) Q^T.
The bi are squared principal stretches, so this takes the logarithm of each
principal stretch and transforms back to the fixed spatial axes. Every subplot
uses the same loading-axis strain (2 hbar12 for shear or hbar11 for tension), not a different strain for
each stress component. The input is prescribed macro F, not a material elastic
gradient or the element's projected F-bar gradient.

`stress_components_vs_strain.svg`: six subplots in order
[11,22,33,12,23,13], each showing ferrite, austenite, and whole-RVE means.

`von_mises_vs_strain.svg`: left, von Mises of each averaged stress tensor;
right, current-volume mean of the local cell von Mises stresses. In both cases
sigma_vm = sqrt(3 J2), with J2 = (s:s)/2 and s = sigma - tr(sigma) I/3.
Tensor contraction counts off-diagonal shear terms twice. Averaging and taking
this invariant do not commute, so the two panels measure different things.
Von Mises is a descriptive stress invariant here, not the TRIP model's yield
criterion. Its values are saved in MPa; J2 itself would have stress-squared units.

`hydrostatic_stress_vs_strain.svg`: current-volume mean hydrostatic Cauchy stress
for each phase and the entire RVE, sigma_h = tr(sigma)/3. Tension is positive;
compression is negative, so pressure would be -sigma_h. The hydrostatic part of
the stress tensor is sigma_h I, complementary to the deviatoric invariant above.
Unlike von Mises stress, trace commutes with averaging: the mean local sigma_h
equals sigma_h computed from the mean stress tensor. Only one curve per region
is therefore needed. These histories are also saved in MPa in the numeric data.

`martensite_vs_strain.svg`: the mean saved `{history['martensite_field']}`
within the initially austenitic region, and its contribution per initial RVE volume.
Both use reference-volume weights; the latter treats initial ferrite as zero.
These summarize the stored SigmaXi variable, not a reconstruction of the exact
current martensite volume fraction accounting for transformation dilatation.
Numeric fractions are 0--1; plots show percentages.

`beta_vs_strain.svg`: reference-volume phase means of the saved beta
outputs (field aliases `beta` or `beta_*`). Beta is the constitutive model's
internal parameter, not an independently defined equivalent plastic strain.

## Periodic kinematic average

The maximum componentwise error of <F_raw>_V0 - Fbar over all saved states is
{history['maximum_volume_average_F_error_inf']:.6e}. Raw F is reconstructed at
the eight quadrature points from reference coordinates and nodal displacement;
integration uses reference-volume weights, not projected material F-bar gradients.
All nine average and prescribed components, per-state errors, and the 3x3 table
of maximum component errors are retained in `history.json`/`history.npz`.

`history.json`: labeled arrays and averaging/unit metadata.
`history.npz`: the same numeric columns for NumPy (`numpy.load`). Saved scalar
state averages, including the selected phase beta outputs, are also included.
All averages use the saved centroid-recovered fields and are postprocessing
approximations; no material library, original properties file, or mesh file is needed.
No source dataset is modified.

Reproduce from the repository root:

```bash
conda run --no-capture-output -n py3.14 python -m verification.summarize_multiphase_trip \\
  PATH_TO_RUN_H5 --output PATH_TO_REPORT_DIRECTORY
```

For renamed transformation output, also supply `--martensite-field FIELD_NAME`.
The same command supports both simple-shear and isochoric-uniaxial databases.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--martensite-field", default="martensite_fraction")
    args = parser.parse_args()
    history = extract_trip_history(args.database, martensite_field=args.martensite_field)
    output = args.output or args.database.parent / "plots"
    write_trip_report(history, output)
    columns = history["columns"]
    component = history["stress_component"]
    print(
        f"saved {len(columns['time_s'])} states to {output}; "
        f"final mean sigma{component}={columns[f'rve_cauchy_stress_{component}_MPa'][-1]:.6g} MPa; "
        f"mean austenite SigmaXi={columns['martensite_mean_initial_austenite_reference'][-1]:.6g}; "
        f"initial-RVE contribution={columns['martensite_contribution_initial_rve_reference'][-1]:.6g}"
    )


if __name__ == "__main__":
    main()
