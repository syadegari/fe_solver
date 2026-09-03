"""Finite-strain finite-element prototype defined by IMPLEMENTATION_SPEC.md."""

from .config import Deck, load_deck
from .preprocess import PreparedAnalysis, prepare_analysis
from .solver import AnalysisResult, run_analysis

__all__ = [
    "AnalysisResult", "Deck", "PreparedAnalysis", "load_deck", "prepare_analysis", "run_analysis",
]
