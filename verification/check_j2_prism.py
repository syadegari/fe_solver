"""Extract and compare qualitative observables for the small J2 square prism."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fe_solver.quadrature import HEX20_POINTS, HEX8_POINTS
from fe_solver.shape import hex20_shape, hex8_shape


SAMPLE_TIMES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class PrismHistory:
    database: str
    analysis_name: str
    formulation: str
    solver_summary: dict[str, Any] | None
    time: np.ndarray
    end_elongation: np.ndarray
    end_reaction_z: np.ndarray
    middle_half_width_x: np.ndarray
    middle_half_width_y: np.ndarray
    minimum_mean_half_width: np.ndarray
    minimum_width_initial_z: np.ndarray
    maximum_equivalent_plastic_strain: np.ndarray
    maximum_plastic_strain_initial_z: np.ndarray
    middle_layer_mean_equivalent_plastic_strain: np.ndarray
    end_layer_mean_equivalent_plastic_strain: np.ndarray
    raw_gauss_J_minimum: np.ndarray
    raw_gauss_J_maximum: np.ndarray
    centroid_J_minimum: np.ndarray
    centroid_J_maximum: np.ndarray
    material_J_minimum: np.ndarray
    material_J_maximum: np.ndarray
    maximum_cross_section_mean_stress_spread: np.ndarray

    def summary(self) -> dict[str, Any]:
        peak = int(np.argmax(np.abs(self.end_reaction_z)))
        final_end_ep = float(self.end_layer_mean_equivalent_plastic_strain[-1])
        localization_ratio = (
            None
            if abs(final_end_ep) <= 1.0e-15
            else float(self.maximum_equivalent_plastic_strain[-1] / final_end_ep)
        )
        initial_width = 0.5 * (
            self.middle_half_width_x[0] + self.middle_half_width_y[0]
        )
        final_width = 0.5 * (
            self.middle_half_width_x[-1] + self.middle_half_width_y[-1]
        )
        return {
            "database": self.database,
            "analysis_name": self.analysis_name,
            "formulation": self.formulation,
            "solver": self.solver_summary,
            "complete_states": int(len(self.time)),
            "final_time": float(self.time[-1]),
            "peak_absolute_end_reaction": float(abs(self.end_reaction_z[peak])),
            "peak_reaction_time": float(self.time[peak]),
            "final_end_reaction_z": float(self.end_reaction_z[-1]),
            "initial_middle_mean_half_width": float(initial_width),
            "final_middle_mean_half_width": float(final_width),
            "middle_width_reduction_fraction": float((initial_width - final_width) / initial_width),
            "final_minimum_mean_half_width": float(self.minimum_mean_half_width[-1]),
            "final_minimum_width_initial_z": float(self.minimum_width_initial_z[-1]),
            "final_maximum_equivalent_plastic_strain": float(
                self.maximum_equivalent_plastic_strain[-1]
            ),
            "final_maximum_plastic_strain_initial_z": float(
                self.maximum_plastic_strain_initial_z[-1]
            ),
            "final_plastic_localization_ratio": localization_ratio,
            "all_raw_gauss_J_range": [
                float(np.min(self.raw_gauss_J_minimum)),
                float(np.max(self.raw_gauss_J_maximum)),
            ],
            "final_raw_gauss_J_range": [
                float(self.raw_gauss_J_minimum[-1]),
                float(self.raw_gauss_J_maximum[-1]),
            ],
            "all_material_J_range": [
                float(np.min(self.material_J_minimum)),
                float(np.max(self.material_J_maximum)),
            ],
            "final_material_J_range": [
                float(self.material_J_minimum[-1]),
                float(self.material_J_maximum[-1]),
            ],
            "final_maximum_cross_section_mean_stress_spread": float(
                self.maximum_cross_section_mean_stress_spread[-1]
            ),
        }

    def curves(self) -> dict[str, list[float]]:
        names = (
            "time",
            "end_elongation",
            "end_reaction_z",
            "middle_half_width_x",
            "middle_half_width_y",
            "minimum_mean_half_width",
            "minimum_width_initial_z",
            "maximum_equivalent_plastic_strain",
            "maximum_plastic_strain_initial_z",
            "middle_layer_mean_equivalent_plastic_strain",
            "end_layer_mean_equivalent_plastic_strain",
            "raw_gauss_J_minimum",
            "raw_gauss_J_maximum",
            "centroid_J_minimum",
            "centroid_J_maximum",
            "material_J_minimum",
            "material_J_maximum",
            "maximum_cross_section_mean_stress_spread",
        )
        return {name: np.asarray(getattr(self, name), dtype=float).tolist() for name in names}

    def as_dict(self, *, include_curves: bool = True) -> dict[str, Any]:
        result = {"summary": self.summary()}
        if include_curves:
            result["curves"] = self.curves()
        return result


def _coordinate_groups(values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    values = np.asarray(values, dtype=float)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(values))))
    order = np.argsort(values)
    groups: list[list[int]] = []
    for index in order:
        if not groups or abs(values[index] - values[groups[-1][0]]) > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    arrays = [np.asarray(group, dtype=np.int64) for group in groups]
    coordinates = np.asarray([np.mean(values[group]) for group in arrays])
    return coordinates, arrays


def _deformation_jacobian_ranges(
    X: np.ndarray,
    displacement: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes_per_element = int(connectivity.shape[1])
    if nodes_per_element == 8:
        quadrature_points = HEX8_POINTS
        shape = hex8_shape
    elif nodes_per_element == 20:
        quadrature_points = HEX20_POINTS
        shape = hex20_shape
    else:
        raise RuntimeError(
            f"small-prism diagnostic does not support {nodes_per_element}-node elements"
        )
    points = [*quadrature_points, np.zeros(3)]
    minimum = np.full(len(displacement), np.inf)
    maximum = np.full(len(displacement), -np.inf)
    center_minimum = np.full(len(displacement), np.inf)
    center_maximum = np.full(len(displacement), -np.inf)
    current = X[None, :, :] + displacement
    for conn in connectivity:
        X_e = X[conn]
        x_e = current[:, conn, :]
        for point_index, point in enumerate(points):
            _, dN = shape(point)
            inverse_J0 = np.linalg.inv(X_e.T @ dN)
            Jx = np.einsum("tni,nj->tij", x_e, dN)
            F = np.einsum("tij,jk->tik", Jx, inverse_J0)
            determinant = np.linalg.det(F)
            if point_index == len(quadrature_points):
                center_minimum = np.minimum(center_minimum, determinant)
                center_maximum = np.maximum(center_maximum, determinant)
            else:
                minimum = np.minimum(minimum, determinant)
                maximum = np.maximum(maximum, determinant)
    return minimum, maximum, center_minimum, center_maximum


def _load_solver_summary(database: Path, history_file: str) -> dict[str, Any] | None:
    path = database.parent / history_file
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        log = json.load(stream)
    increments = log.get("increments", [])
    newton = log.get("newton_history", [])
    if not increments:
        return {
            "run_log": str(path),
            "execution": log.get("execution", {}),
            "timing": log.get("timing", {}),
            "accepted_increments": 0,
        }
    step_sizes = [float(row["t_np1"]) - float(row["t_n"]) for row in increments]
    failed_lines = [
        row
        for row in newton
        if row.get("line_search_trials") is not None and row.get("line_search_alpha") is None
    ]
    return {
        "run_log": str(path),
        "execution": log.get("execution", {}),
        "timing": log.get("timing", {}),
        "logged_start_time": float(increments[0]["t_n"]),
        "logged_final_time": float(increments[-1]["t_np1"]),
        "accepted_increments": len(increments),
        "newton_records": len(newton),
        "maximum_converged_newton_iterations": max(
            int(row["newton_iterations"]) for row in increments
        ),
        "maximum_attempts_for_accepted_increment": max(int(row["attempts"]) for row in increments),
        "total_cutbacks": sum(int(row["cutbacks"]) for row in increments),
        "accepted_step_size_range": [min(step_sizes), max(step_sizes)],
        "failed_line_searches": len(failed_lines),
        "total_line_search_trials": sum(int(row.get("line_search_trials", 0)) for row in newton),
        "total_line_search_backtracks": sum(
            int(row.get("line_search_backtracks", 0)) for row in newton
        ),
    }


def extract_prism_history(database: str | Path) -> PrismHistory:
    path = Path(database)
    with h5py.File(path, "r") as archive:
        complete = int(archive["results"].attrs["n_complete_steps"])
        if complete < 1:
            raise RuntimeError("prism database has no complete result states")
        blocks = sorted(archive["mesh/blocks"])
        if blocks != ["0000"]:
            raise RuntimeError("small-prism diagnostic requires exactly one element block")
        mesh_block = archive["mesh/blocks/0000"]
        formulation = str(mesh_block.attrs["formulation"])
        if formulation not in ("hex8", "hex8_fbar", "hex20"):
            raise RuntimeError("small-prism diagnostic requires Hex8, Hex8-Fbar, or Hex20")
        result_block = archive["results/blocks/0000"]
        state = result_block["state"]
        if "equivalent_plastic_strain" not in state:
            raise RuntimeError("prism database has no equivalent plastic strain")
        time = np.asarray(archive["results/time"][:complete], dtype=float)
        X = np.asarray(archive["mesh/reference_coordinates"], dtype=float)
        displacement = np.asarray(
            archive["results/nodal/displacement"][:complete], dtype=float
        )
        reaction = np.asarray(
            archive["results/nodal/constraint_reaction"][:complete], dtype=float
        )
        connectivity = np.asarray(mesh_block["connectivity"], dtype=np.int64)
        equivalent_plastic_strain = np.asarray(
            state["equivalent_plastic_strain"][:complete], dtype=float
        )
        stress = np.asarray(result_block["cauchy_stress"][:complete], dtype=float)
        analysis_name = str(archive["meta"].attrs["analysis_name"])
        resolved_input = archive["meta/resolved_input_json"][()]
        if isinstance(resolved_input, bytes):
            resolved_input = resolved_input.decode("utf-8")
        history_file = str(json.loads(str(resolved_input))["output"]["history_file"])

    if equivalent_plastic_strain.shape != (len(time), len(connectivity)):
        raise RuntimeError("equivalent plastic strain has an unexpected shape")
    z_nodes, node_layers = _coordinate_groups(X[:, 2])
    center_shape = hex8_shape if connectivity.shape[1] == 8 else hex20_shape
    center_weights, _ = center_shape(np.zeros(3))
    centers = np.einsum("a,eai->ei", center_weights, X[connectivity])
    z_elements, element_layers = _coordinate_groups(centers[:, 2])
    current = X[None, :, :] + displacement
    width_x = np.empty((len(time), len(node_layers)))
    width_y = np.empty_like(width_x)
    for layer_index, nodes in enumerate(node_layers):
        width_x[:, layer_index] = np.ptp(current[:, nodes, 0], axis=1)
        width_y[:, layer_index] = np.ptp(current[:, nodes, 1], axis=1)
    mean_width = 0.5 * (width_x + width_y)
    minimum_layer = np.argmin(mean_width, axis=1)

    middle_nodes = node_layers[0]
    end_nodes = node_layers[-1]
    end_elongation = np.mean(displacement[:, end_nodes, 2], axis=1)
    end_reaction = np.sum(reaction[:, end_nodes, 2], axis=1)
    maximum_ep_element = np.argmax(equivalent_plastic_strain, axis=1)
    middle_ep = np.mean(equivalent_plastic_strain[:, element_layers[0]], axis=1)
    end_ep = np.mean(equivalent_plastic_strain[:, element_layers[-1]], axis=1)

    mean_stress = (stress[..., 0] + stress[..., 1] + stress[..., 2]) / 3.0
    stress_spread = np.zeros(len(time))
    for elements in element_layers:
        stress_spread = np.maximum(
            stress_spread,
            np.ptp(mean_stress[:, elements], axis=1),
        )

    raw_min, raw_max, center_min, center_max = _deformation_jacobian_ranges(
        X, displacement, connectivity
    )
    material_min = center_min if formulation == "hex8_fbar" else raw_min
    material_max = center_max if formulation == "hex8_fbar" else raw_max
    return PrismHistory(
        str(path),
        analysis_name,
        formulation,
        _load_solver_summary(path, history_file),
        time,
        end_elongation,
        end_reaction,
        width_x[:, 0],
        width_y[:, 0],
        mean_width[np.arange(len(time)), minimum_layer],
        z_nodes[minimum_layer],
        np.max(equivalent_plastic_strain, axis=1),
        centers[maximum_ep_element, 2],
        middle_ep,
        end_ep,
        raw_min,
        raw_max,
        center_min,
        center_max,
        material_min,
        material_max,
        stress_spread,
    )


def _interpolate(history: PrismHistory, name: str, times: np.ndarray) -> np.ndarray:
    return np.interp(times, history.time, np.asarray(getattr(history, name), dtype=float))


def compare_prism_histories(
    reference: PrismHistory,
    candidate: PrismHistory,
) -> dict[str, Any]:
    common_end = min(float(reference.time[-1]), float(candidate.time[-1]))
    sample_times = np.asarray([time for time in SAMPLE_TIMES if time <= common_end + 1.0e-12])
    if not len(sample_times) or sample_times[-1] < common_end - 1.0e-12:
        sample_times = np.append(sample_times, common_end)
    fields = (
        "end_reaction_z",
        "middle_half_width_x",
        "middle_half_width_y",
        "maximum_equivalent_plastic_strain",
        "raw_gauss_J_minimum",
        "raw_gauss_J_maximum",
        "material_J_minimum",
        "material_J_maximum",
        "maximum_cross_section_mean_stress_spread",
    )
    samples: list[dict[str, Any]] = []
    for index, time in enumerate(sample_times):
        row: dict[str, Any] = {"time": float(time), "fields": {}}
        for name in fields:
            reference_value = float(_interpolate(reference, name, sample_times)[index])
            candidate_value = float(_interpolate(candidate, name, sample_times)[index])
            scale = max(abs(reference_value), 1.0e-15)
            row["fields"][name] = {
                "reference": reference_value,
                "candidate": candidate_value,
                "candidate_minus_reference": candidate_value - reference_value,
                "relative_difference": (candidate_value - reference_value) / scale,
            }
        samples.append(row)

    def peak_before(history: PrismHistory) -> tuple[float, float]:
        eligible = np.flatnonzero(history.time <= common_end + 1.0e-12)
        local = int(np.argmax(np.abs(history.end_reaction_z[eligible])))
        index = int(eligible[local])
        return float(history.time[index]), float(abs(history.end_reaction_z[index]))

    reference_peak_time, reference_peak = peak_before(reference)
    candidate_peak_time, candidate_peak = peak_before(candidate)
    return {
        "reference_database": reference.database,
        "reference_formulation": reference.formulation,
        "candidate_database": candidate.database,
        "candidate_formulation": candidate.formulation,
        "common_end_time": common_end,
        "reference_peak": {"time": reference_peak_time, "absolute_reaction": reference_peak},
        "candidate_peak": {"time": candidate_peak_time, "absolute_reaction": candidate_peak},
        "candidate_to_reference_peak_reaction_ratio": candidate_peak / max(reference_peak, 1.0e-15),
        "samples": samples,
    }


def plot_prism_histories(
    reference: PrismHistory,
    candidate: PrismHistory,
    output: str | Path,
) -> Path:
    """Plot physical response and volumetric diagnostics for a formulation pair."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    histories = (reference, candidate)
    labels = {
        "hex8_fbar": "Hex8-Fbar",
        "hex8": "Hex8",
        "hex20": "Hex20",
    }
    colors = {
        "hex8_fbar": "#1f77b4",
        "hex8": "#d95f02",
        "hex20": "#2ca02c",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True, constrained_layout=True)

    for history in histories:
        label = labels.get(history.formulation, history.formulation)
        linestyle = "-"
        if "refined" in history.analysis_name:
            label += " refined 4x4x48"
            linestyle = "--"
        elif "soft_bulk" in history.analysis_name:
            label += r" $K/2$"
            linestyle = ":"
        color = colors.get(history.formulation)
        elongation = history.end_elongation
        mean_half_width = 0.5 * (
            history.middle_half_width_x + history.middle_half_width_y
        )
        axes[0, 0].plot(
            elongation, np.abs(history.end_reaction_z) / 1000.0,
            color=color, linestyle=linestyle, label=label,
        )
        axes[0, 1].plot(
            elongation, mean_half_width, color=color, linestyle=linestyle, label=label
        )
        axes[0, 2].plot(
            elongation, history.maximum_equivalent_plastic_strain,
            color=color, linestyle=linestyle, label=label,
        )
        axes[1, 0].plot(
            elongation, history.maximum_cross_section_mean_stress_spread,
            color=color, linestyle=linestyle, label=label,
        )
        axes[1, 1].fill_between(
            elongation, history.raw_gauss_J_minimum, history.raw_gauss_J_maximum,
            color=color, alpha=0.22, label=label,
        )
        axes[1, 1].plot(
            elongation, history.raw_gauss_J_minimum,
            color=color, linestyle=linestyle, linewidth=0.8,
        )
        axes[1, 1].plot(
            elongation, history.raw_gauss_J_maximum,
            color=color, linestyle=linestyle, linewidth=0.8,
        )
        axes[1, 2].fill_between(
            elongation, history.material_J_minimum, history.material_J_maximum,
            color=color, alpha=0.22, label=label,
        )
        axes[1, 2].plot(
            elongation, history.material_J_minimum,
            color=color, linestyle=linestyle, linewidth=0.8,
        )
        axes[1, 2].plot(
            elongation, history.material_J_maximum,
            color=color, linestyle=linestyle, linewidth=0.8,
        )

    axes[0, 0].set(ylabel=r"absolute end reaction $|R_z|$ [kN]", title="Force response")
    axes[0, 1].set(ylabel="middle mean half-width [mm]", title="Neck contraction")
    axes[0, 2].set(ylabel="maximum equivalent plastic strain", title="Plastic localization")
    axes[1, 0].set(
        xlabel="prescribed end elongation [mm]",
        ylabel="maximum section mean-stress spread [MPa]",
        title="Hydrostatic-stress variation",
    )
    axes[1, 1].set(
        xlabel="prescribed end elongation [mm]",
        ylabel=r"raw Gauss-point $J$ range",
        title="Kinematic volume ratio",
    )
    axes[1, 2].set(
        xlabel="prescribed end elongation [mm]",
        ylabel=r"material-seen $J$ range",
        title="Constitutive volume ratio",
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend()
    common_time = min(float(history.time[-1]) for history in histories)
    common_elongation = min(
        float(np.interp(common_time, history.time, history.end_elongation))
        for history in histories
    )
    for axis in axes.flat:
        axis.set_xlim(left=0.0, right=common_elongation)
    axes[1, 1].axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    axes[1, 2].axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    fig.suptitle("Small J2 necking prism: standard Hex8 versus centroidal F-bar")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="extract or compare small J2 square-prism observables"
    )
    parser.add_argument("database", type=Path, help="reference or single run.h5")
    parser.add_argument("--compare", type=Path, default=None, help="candidate run.h5")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="write a six-panel PNG comparison (requires --compare)",
    )
    args = parser.parse_args(argv)
    if args.plot is not None and args.compare is None:
        parser.error("--plot requires --compare")
    reference = extract_prism_history(args.database)
    if args.compare is None:
        report = reference.as_dict(include_curves=not args.summary_only)
    else:
        candidate = extract_prism_history(args.compare)
        report = {
            "reference": reference.as_dict(include_curves=not args.summary_only),
            "candidate": candidate.as_dict(include_curves=not args.summary_only),
            "comparison": compare_prism_histories(reference, candidate),
        }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    if args.plot is not None:
        assert args.compare is not None
        plot_prism_histories(reference, candidate, args.plot)


if __name__ == "__main__":
    main()
