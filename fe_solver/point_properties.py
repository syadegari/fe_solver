"""Load immutable per-element properties from solver-neutral HDF5 sidecars."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import Deck
from .types import ModelError, PointProperties


POINT_PROPERTY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PointPropertySource:
    name: str
    path: Path
    property_names: tuple[str, ...]
    by_element_tag: dict[int, PointProperties]


def _immutable_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    result = array.copy()
    result.flags.writeable = False
    return result


def load_point_property_sources(deck: Deck) -> dict[str, PointPropertySource]:
    sources: dict[str, PointPropertySource] = {}
    for entry in deck.data.get("point_property_sources", []):
        if not isinstance(entry, dict):
            raise ModelError("each point_property_sources entry must be a table")
        unknown = set(entry) - {"name", "file", "format"}
        if unknown:
            raise ModelError(f"unknown point-property source fields: {sorted(unknown)}")
        missing = {"name", "file"} - set(entry)
        if missing:
            raise ModelError(f"point-property source is missing fields: {sorted(missing)}")
        name = str(entry["name"])
        if not name or name in sources:
            message = "empty" if not name else f"duplicate {name!r}"
            raise ModelError(f"{message} point-property source name")
        source_format = str(entry.get("format", "hdf5_element_properties"))
        if source_format != "hdf5_element_properties":
            raise ModelError(
                f"point-property source {name!r} has unsupported format {source_format!r}"
            )
        path = deck.resolve(str(entry["file"]))
        try:
            archive = h5py.File(path, "r")
        except OSError as exc:
            raise ModelError(f"cannot read point-property source {path}") from exc
        with archive:
            if int(archive.attrs.get("schema_version", -1)) != POINT_PROPERTY_SCHEMA_VERSION:
                raise ModelError(
                    f"point-property source {name!r} has unsupported schema version"
                )
            if "element_tags" not in archive or "properties" not in archive:
                raise ModelError(
                    f"point-property source {name!r} requires element_tags and properties"
                )
            tags = np.asarray(archive["element_tags"])
            if tags.ndim != 1 or not np.issubdtype(tags.dtype, np.integer):
                raise ModelError(
                    f"point-property source {name!r} element_tags must be a one-dimensional integer array"
                )
            tags = tags.astype(np.int64, copy=False)
            if len(tags) != len(np.unique(tags)):
                raise ModelError(f"point-property source {name!r} has duplicate element tags")
            group = archive["properties"]
            if not isinstance(group, h5py.Group) or not group:
                raise ModelError(f"point-property source {name!r} has no properties")
            property_names = tuple(sorted(group.keys()))
            values: dict[str, np.ndarray] = {}
            for property_name in property_names:
                if not property_name.isidentifier():
                    raise ModelError(
                        f"invalid point-property name {property_name!r} in source {name!r}"
                    )
                dataset = group[property_name]
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 1:
                    raise ModelError(
                        f"point property {property_name!r} in source {name!r} must be a dataset"
                    )
                if dataset.shape[0] != len(tags):
                    raise ModelError(
                        f"point property {property_name!r} in source {name!r} has the wrong row count"
                    )
                if not (
                    np.issubdtype(dataset.dtype, np.integer)
                    or np.issubdtype(dataset.dtype, np.floating)
                    or np.issubdtype(dataset.dtype, np.bool_)
                ):
                    raise ModelError(
                        f"point property {property_name!r} in source {name!r} must be numeric"
                    )
                data = np.asarray(dataset)
                if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
                    raise ModelError(
                        f"point property {property_name!r} in source {name!r} contains non-finite values"
                    )
                values[property_name] = data
            by_element_tag = {
                int(tag): PointProperties(
                    tuple(
                        (property_name, _immutable_value(values[property_name][row]))
                        for property_name in property_names
                    )
                )
                for row, tag in enumerate(tags)
            }
        sources[name] = PointPropertySource(
            name, path, property_names, by_element_tag
        )
    return sources
