"""Compact, solver-facing startup and linear-system reporting."""
from __future__ import annotations

from .assembly import FEModel


def build_analysis_summary(
    model: FEModel,
    *,
    analysis_start: float,
    analysis_end: float,
    run_start: float,
    run_target: float,
    restarted: bool,
    solution_method: str,
    solution_backend: str,
    unknowns_by_type: dict[str, int],
    solution_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the model-size information known before solution begins."""
    formulation_counts: dict[str, int] = {}
    for block in model.blocks:
        formulation_counts[block.formulation] = (
            formulation_counts.get(block.formulation, 0) + len(block.connectivity)
        )
    unknowns = {name: int(count) for name, count in unknowns_by_type.items()}
    return {
        "name": str(model.deck.data["analysis"]["name"]),
        "time": {
            "analysis_start": analysis_start,
            "analysis_end": analysis_end,
            "run_start": run_start,
            "run_target": run_target,
            "restarted": restarted,
        },
        "mesh": {
            "nodes": int(len(model.mesh.X)),
            "elements": int(sum(formulation_counts.values())),
            "elements_by_formulation": dict(sorted(formulation_counts.items())),
        },
        "linear_system": {
            "method": solution_method,
            "backend": solution_backend,
            "options": solution_options or {},
            "unknowns": {
                "by_type": unknowns,
                "total": sum(unknowns.values()),
            },
        },
    }


def print_startup_summary(summary: dict[str, object]) -> None:
    """Print only the durable model-size portion of the analysis summary."""
    time = summary["time"]
    mesh = summary["mesh"]
    restart_label = "restart" if time["restarted"] else "cold start"
    formulations = ", ".join(
        f"{name}={count}"
        for name, count in mesh["elements_by_formulation"].items()
    )
    print(
        f"analysis: {summary['name']}; {restart_label} at t={float(time['run_start']):.12g}, "
        f"target t={float(time['run_target']):.12g}"
    )
    print(
        f"model: {mesh['nodes']} nodes, {mesh['elements']} elements "
        f"({formulations})"
    )


def record_sparse_system(
    summary: dict[str, object],
    *,
    primary_name: str,
    primary_shape: tuple[int, int],
    primary_stored_entries: int,
    system_name: str,
    system_shape: tuple[int, int],
    system_stored_entries: int,
) -> None:
    """Attach storage data supplied by whichever global solver assembled it."""
    dense_entries = int(system_shape[0] * system_shape[1])
    summary["linear_system"]["matrices"] = {
        "primary": {
            "name": primary_name,
            "shape": list(primary_shape),
            "stored_entries": int(primary_stored_entries),
        },
        "system": {
            "name": system_name,
            "shape": list(system_shape),
            "stored_entries": int(system_stored_entries),
            "dense_entry_count": dense_entries,
            "storage_fraction": (
                float(system_stored_entries / dense_entries) if dense_entries else 0.0
            ),
        },
    }


def print_linear_system_summary(summary: dict[str, object]) -> None:
    linear = summary["linear_system"]
    unknowns = linear["unknowns"]
    matrices = linear["matrices"]
    primary = matrices["primary"]
    system = matrices["system"]
    unknown_description = " + ".join(
        f"{count} {name.replace('_', ' ')}"
        for name, count in unknowns["by_type"].items()
    )
    print(
        f"linear system: {linear['backend']} {system['name']}; "
        f"{primary['name']} {primary['shape'][0]} x {primary['shape'][1]} with "
        f"{primary['stored_entries']} stored entries; "
        f"{system['name']} {system['shape'][0]} x {system['shape'][1]} "
        f"({unknown_description} unknowns) with "
        f"{system['stored_entries']}/{system['dense_entry_count']} stored entries "
        f"({100.0 * float(system['storage_fraction']):.2f}%)"
    )
