from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

from .types import ModelError


@dataclass(frozen=True)
class Curve:
    name: str
    times: np.ndarray
    values: np.ndarray

    def evaluate(self, t: float) -> float:
        tol = 1.0e-12 * max(1.0, abs(t), abs(float(self.times[0])), abs(float(self.times[-1])))
        if t < self.times[0] - tol or t > self.times[-1] + tol:
            raise ModelError(f"curve {self.name!r} is not defined at pseudo-time {t}")
        return float(np.interp(np.clip(t, self.times[0], self.times[-1]), self.times, self.values))


@dataclass(frozen=True)
class ValueExpression:
    constant: float = 0.0
    curve: str | None = None
    scale: float = 1.0

    def evaluate(self, t: float, curves: dict[str, Curve]) -> float:
        value = self.constant
        if self.curve is not None:
            try:
                value += self.scale * curves[self.curve].evaluate(t)
            except KeyError as exc:
                raise ModelError(f"unknown curve {self.curve!r}") from exc
        return float(value)


def value_expression(data: object) -> ValueExpression:
    if isinstance(data, (int, float)):
        return ValueExpression(float(data))
    if not isinstance(data, dict):
        raise ModelError(f"invalid scalar value expression: {data!r}")
    unknown = set(data) - {"constant", "curve", "scale"}
    if unknown:
        raise ModelError(f"unknown value-expression fields: {sorted(unknown)}")
    if "constant" not in data and "curve" not in data:
        raise ModelError("a value expression requires a constant or curve term")
    return ValueExpression(
        float(data.get("constant", 0.0)),
        str(data["curve"]) if "curve" in data else None,
        float(data.get("scale", 1.0)),
    )


@dataclass(frozen=True)
class Deck:
    path: Path
    data: dict
    curves: dict[str, Curve]

    @property
    def root(self) -> Path:
        return self.path.parent

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p


def load_deck(path: str | Path) -> Deck:
    deck_path = Path(path).resolve()
    with deck_path.open("rb") as stream:
        data = tomllib.load(stream)
    required = {
        "analysis", "mesh", "time", "nonlinear", "linear_solver", "materials",
        "element_assignments", "output", "restart", "verification",
    }
    missing = required - set(data)
    if missing:
        raise ModelError(f"input deck is missing tables: {sorted(missing)}")
    analysis = data["analysis"]
    t0, t1 = float(analysis["t_start"]), float(analysis["t_end"])
    if not t1 > t0:
        raise ModelError("analysis.t_end must be greater than analysis.t_start")
    curves: dict[str, Curve] = {}
    for entry in data.get("curves", []):
        name = str(entry["name"])
        if name in curves:
            raise ModelError(f"duplicate curve name {name!r}")
        points = np.asarray(entry["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ModelError(f"curve {name!r} requires at least two [time, value] points")
        if not np.all(np.diff(points[:, 0]) > 0.0):
            raise ModelError(f"curve {name!r} times must be strictly increasing")
        if points[0, 0] > t0 or points[-1, 0] < t1:
            raise ModelError(f"curve {name!r} does not cover the analysis interval")
        curves[name] = Curve(name, points[:, 0].copy(), points[:, 1].copy())
    for table in data.get("constraints", {}).get("prescribed", []):
        value_expression(table["value"])
    return Deck(deck_path, data, curves)


def component_index(component: str) -> int:
    try:
        return {"x": 0, "y": 1, "z": 2}[component]
    except KeyError as exc:
        raise ModelError(f"unknown displacement component {component!r}") from exc


def uniform_events(t_start: float, t_end: float, interval: float | None) -> list[float]:
    if interval is None:
        return []
    interval = float(interval)
    if interval <= 0.0:
        raise ModelError("event interval must be positive")
    count = int(np.floor((t_end - t_start) / interval + 1.0e-12))
    return [t_start + k * interval for k in range(1, count + 1) if t_start + k * interval <= t_end]


def mandatory_events(deck: Deck) -> tuple[np.ndarray, set[float], set[float]]:
    data = deck.data
    t0 = float(data["analysis"]["t_start"])
    t1 = float(data["analysis"]["t_end"])
    output = data["output"]
    restart = data["restart"]
    output_times = uniform_events(t0, t1, output.get("interval")) + [float(x) for x in output.get("explicit_times", [])]
    restart_times = []
    if restart.get("enabled", False):
        restart_times = uniform_events(t0, t1, restart.get("interval")) + [float(x) for x in restart.get("explicit_times", [])]
    candidates = [t1, *output_times, *restart_times]
    for curve in deck.curves.values():
        candidates.extend(float(x) for x in curve.times if t0 < x <= t1)
    tol = 1.0e-12 * max(1.0, abs(t0), abs(t1))
    merged: list[float] = []
    for value in sorted(x for x in candidates if t0 < x <= t1 + tol):
        value = min(value, t1)
        if not merged or abs(value - merged[-1]) > tol:
            merged.append(value)
    def snap_set(values: list[float]) -> set[float]:
        return {next((event for event in merged if abs(event - value) <= tol), value) for value in values}
    return np.asarray(merged), snap_set(output_times), snap_set(restart_times)

