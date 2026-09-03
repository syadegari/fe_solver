"""Finite-strain finite-element prototype defined by IMPLEMENTATION_SPEC.md."""

from .config import Deck, load_deck
from .solver import AnalysisResult, run_analysis

__all__ = ["AnalysisResult", "Deck", "load_deck", "run_analysis"]
