from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assembly import FEModel, build_model
from .config import Deck, load_deck, mandatory_events
from .constraints import ConstraintSystem, build_constraints
from .mesh import Mesh, read_gmsh
from .types import ModelError


@dataclass(frozen=True)
class PreparedAnalysis:
    deck: Deck
    mesh: Mesh
    model: FEModel
    constraints: ConstraintSystem
    events: np.ndarray
    output_times: set[float]
    restart_times: set[float]


def prepare_analysis(deck_or_path: Deck | str | Path) -> PreparedAnalysis:
    """Read and validate all inputs needed before nonlinear solution begins."""
    deck = deck_or_path if isinstance(deck_or_path, Deck) else load_deck(deck_or_path)
    if deck.data["mesh"].get("format") != "gmsh_msh41":
        raise ModelError("v1 supports only mesh.format = 'gmsh_msh41'")
    if deck.data["linear_solver"].get("backend") not in {
        "scipy_splu",
        "superlu_mt",
    }:
        raise ModelError(
            "linear_solver.backend must be 'scipy_splu' or 'superlu_mt'"
        )
    mesh = read_gmsh(deck.resolve(str(deck.data["mesh"]["file"])))
    model = build_model(deck, mesh)
    constraints = build_constraints(deck, mesh)
    events, output_times, restart_times = mandatory_events(deck)
    return PreparedAnalysis(deck, mesh, model, constraints, events, output_times, restart_times)
