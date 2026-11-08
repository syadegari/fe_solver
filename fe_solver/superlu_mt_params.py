"""Expert-level SuperLU_MT factorization knobs and their provenance.

These parameters are deliberately *not* part of the ordinary analysis deck
contract.  A normal run never mentions them and silently receives the built-in
defaults, which are defined here once and nowhere else.  A user who needs to
deviate must opt in explicitly, and every deviation is reported at startup and
recorded in the run summary, so a tuned run can never be mistaken for a default
one.

Resolution order, least to most specific:

    built-in defaults  <-  expert parameter file  <-  deck [linear_solver.superlu_mt]

``num_threads`` is intentionally absent.  Thread count belongs to the person
running the experiment, not to the person preparing the deck, so it stays on the
solver command line only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math
import tomllib

from .types import ModelError


#: The single source of truth for SuperLU_MT knob defaults.
#:
#: ``panel_size`` and ``relax`` of ``-1`` instruct the native bridge to query
#: ``sp_ienv(1)`` and ``sp_ienv(2)``, i.e. SuperLU_MT's own tuned defaults, so the
#: library stays the authority for them rather than this file.
DEFAULTS: dict[str, float | int] = {
    "diag_pivot_thresh": 1.0,
    "panel_size": -1,
    "relax": -1,
    "fill_lusup": 0,
    "fill_ucol": 0,
    "fill_lsub": 0,
    "max_supernode": -1,
    "rowblk": -1,
    "colblk": -1,
}

#: Blocking parameters served by the bridge's own sp_ienv() override.  They
#: trade dense-kernel efficiency against cache residency and thread
#: granularity, are coupled to one another, and should be tuned together
#: against measured factorization time.  -1 keeps the library default.
BLOCKING_PARAMS = ("max_supernode", "rowblk", "colblk")

#: The factor-storage estimates share sp_ienv(6/7/8) semantics: negative is a
#: fill-growth factor on nnz(A), positive is an absolute entry count, and the
#: default 0 keeps the library's own sp_ienv value (-50 / -50 / -30).  They
#: exist because undersized factor storage is fatal: SuperLU_MT exits the whole
#: process when the factors outgrow these arrays mid-factorization.  See
#: docs/SUPERLU_MT_INTERNALS.md.
FILL_PARAMS = ("fill_lusup", "fill_ucol", "fill_lsub")

_INT32_MAX = 2**31 - 1

#: TOML table that carries the knobs, both in the expert file and in the deck.
PARAMS_TABLE = "superlu_mt"

_DESCRIPTIONS: dict[str, str] = {
    "diag_pivot_thresh": "diagonal pivoting threshold; 1.0 = partial pivoting",
    "panel_size": "columns per panel; -1 = SuperLU_MT sp_ienv(1)",
    "relax": "relaxed supernode size; -1 = SuperLU_MT sp_ienv(2)",
    "fill_lusup": "L supernode value storage; <0 factor on nnz(A), >0 entries, 0 = sp_ienv(6)",
    "fill_ucol": "U column storage; <0 factor on nnz(A), >0 entries, 0 = sp_ienv(7)",
    "fill_lsub": "L subscript storage; <0 factor on nnz(A), >0 entries, 0 = sp_ienv(8)",
    "max_supernode": "max columns per supernode; -1 = SuperLU_MT sp_ienv(3)",
    "rowblk": "2-D blocking row minimum; -1 = SuperLU_MT sp_ienv(4)",
    "colblk": "2-D blocking column minimum; -1 = SuperLU_MT sp_ienv(5)",
}


def _validate(name: str, value: object) -> float | int:
    """Mirror the native bridge's own range checks, with deck-level wording."""
    if isinstance(value, bool):
        raise ModelError(f"superlu_mt.{name} must be a number, got a boolean")
    if name == "diag_pivot_thresh":
        if not isinstance(value, (int, float)):
            raise ModelError(f"superlu_mt.{name} must be a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ModelError(
                f"superlu_mt.{name} must be a finite value in [0, 1], got {value!r}"
            )
        return value
    if not isinstance(value, int):
        raise ModelError(f"superlu_mt.{name} must be an integer, got {value!r}")
    if name == "panel_size" and not (value == -1 or value >= 1):
        raise ModelError(
            f"superlu_mt.{name} must be -1 (library default) or at least 1, got {value}"
        )
    if name == "relax" and value < -1:
        raise ModelError(
            f"superlu_mt.{name} must be -1 (library default) or at least 0, got {value}"
        )
    if name in BLOCKING_PARAMS and not (value == -1 or value >= 1):
        raise ModelError(
            f"superlu_mt.{name} must be -1 (library default) or at least 1, got {value}"
        )
    if name in FILL_PARAMS and abs(value) > _INT32_MAX:
        raise ModelError(
            f"superlu_mt.{name} must fit a 32-bit integer, got {value}"
        )
    return value


def _validate_table(table: object, origin: str) -> dict[str, float | int]:
    """Accept only known keys; a typo must never fall back to a default."""
    if not isinstance(table, dict):
        raise ModelError(f"{origin} must be a table of SuperLU_MT parameters")
    unknown = set(table) - set(DEFAULTS)
    if unknown:
        raise ModelError(
            f"unknown SuperLU_MT parameters in {origin}: {sorted(unknown)}; "
            f"supported parameters are {sorted(DEFAULTS)}"
        )
    return {name: _validate(name, value) for name, value in table.items()}


@dataclass(frozen=True)
class SuperLUMTParams:
    """Effective SuperLU_MT knobs together with where each value came from."""

    diag_pivot_thresh: float = DEFAULTS["diag_pivot_thresh"]
    panel_size: int = DEFAULTS["panel_size"]
    relax: int = DEFAULTS["relax"]
    fill_lusup: int = DEFAULTS["fill_lusup"]
    fill_ucol: int = DEFAULTS["fill_ucol"]
    fill_lsub: int = DEFAULTS["fill_lsub"]
    max_supernode: int = DEFAULTS["max_supernode"]
    rowblk: int = DEFAULTS["rowblk"]
    colblk: int = DEFAULTS["colblk"]
    #: Parameter name -> "default", "file" or "deck".
    sources: dict[str, str] = field(
        default_factory=lambda: {name: "default" for name in DEFAULTS}
    )
    #: Path of the expert file, recorded even when every value it set was
    #: subsequently overridden by the deck.
    params_file: Path | None = None
    params_file_sha256: str | None = None

    @property
    def overrides(self) -> dict[str, float | int]:
        """Parameters whose effective value actually deviates from the default.

        Membership is decided by value, not by provenance: a file or deck that
        explicitly restates a default is not a deviation and must not raise a
        false alarm.  ``sources`` still records where the value was written.
        """
        return {
            name: getattr(self, name)
            for name in DEFAULTS
            if getattr(self, name) != DEFAULTS[name]
        }

    def factor_kwargs(self) -> dict[str, float | int]:
        """Keyword arguments for :meth:`SuperLUMTFactor.factor`."""
        return {name: getattr(self, name) for name in DEFAULTS}

    def summary_options(self) -> dict[str, object]:
        """Provenance block persisted under ``analysis['linear_system']``."""
        options: dict[str, object] = {
            "values": self.factor_kwargs(),
            "sources": dict(self.sources),
            "defaults": dict(DEFAULTS),
            "overridden": sorted(self.overrides),
        }
        if self.params_file is not None:
            options["params_file"] = str(self.params_file)
            options["params_file_sha256"] = self.params_file_sha256
        return options


def load_params_file(path: str | Path) -> tuple[dict[str, float | int], Path, str]:
    """Read and validate an expert parameter file, returning its content and hash."""
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ModelError(f"cannot read SuperLU_MT parameter file {resolved}: {exc}") from exc
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ModelError(f"invalid TOML in SuperLU_MT parameter file {resolved}: {exc}") from exc
    unknown = set(document) - {PARAMS_TABLE}
    if unknown:
        raise ModelError(
            f"SuperLU_MT parameter file {resolved} has unexpected tables: {sorted(unknown)}; "
            f"all parameters belong under [{PARAMS_TABLE}]"
        )
    table = _validate_table(
        document.get(PARAMS_TABLE, {}), f"[{PARAMS_TABLE}] in {resolved}"
    )
    return table, resolved, hashlib.sha256(raw).hexdigest()


def resolve_params(
    deck_table: object = None,
    params_file: str | Path | None = None,
) -> SuperLUMTParams:
    """Layer defaults, the expert file and the deck into effective parameters."""
    values = dict(DEFAULTS)
    sources = {name: "default" for name in DEFAULTS}
    file_path: Path | None = None
    file_digest: str | None = None

    if params_file is not None:
        from_file, file_path, file_digest = load_params_file(params_file)
        for name, value in from_file.items():
            values[name] = value
            sources[name] = "file"

    if deck_table is not None:
        from_deck = _validate_table(deck_table, f"deck [linear_solver.{PARAMS_TABLE}]")
        for name, value in from_deck.items():
            values[name] = value
            sources[name] = "deck"

    return SuperLUMTParams(
        **values,
        sources=sources,
        params_file=file_path,
        params_file_sha256=file_digest,
    )


def format_override_banner(params: SuperLUMTParams) -> list[str]:
    """Render the startup banner; empty when every value is a default."""
    overrides = params.overrides
    if not overrides and params.params_file is None:
        return []
    lines = ["*" * 72]
    if overrides:
        lines.append(
            f"*** EXPERT SuperLU_MT OVERRIDES ACTIVE "
            f"({len(overrides)} of {len(DEFAULTS)} parameters) ***"
        )
    else:
        lines.append(
            "*** SuperLU_MT parameter file loaded; no parameter differs from defaults ***"
        )
    if params.params_file is not None:
        lines.append(f"    parameter file: {params.params_file}")
        lines.append(f"    sha256:         {params.params_file_sha256}")
    for name in DEFAULTS:
        value = getattr(params, name)
        source = params.sources.get(name, "default")
        if name not in overrides:
            restated = "" if source == "default" else f", restated by {source}"
            lines.append(f"    {name} = {value!r} (default{restated})")
        else:
            lines.append(
                f"    {name} = {value!r}  <-- OVERRIDDEN by {source} "
                f"(default {DEFAULTS[name]!r}); {_DESCRIPTIONS[name]}"
            )
    if float(params.diag_pivot_thresh) < 1.0:
        lines.append(
            "    WARNING: diag_pivot_thresh < 1.0 relaxes pivoting and trades "
            "numerical accuracy for speed."
        )
    positive_fills = [
        name for name in FILL_PARAMS if int(getattr(params, name)) > 0
    ]
    if positive_fills:
        lines.append(
            f"    NOTE: positive {', '.join(positive_fills)} are ABSOLUTE entry "
            "counts, not growth factors; an undersized value aborts the process "
            "mid-factorization."
        )
    lines.append("*" * 72)
    return lines


def print_override_banner(params: SuperLUMTParams) -> None:
    for line in format_override_banner(params):
        print(line)
