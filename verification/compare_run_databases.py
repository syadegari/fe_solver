from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


IGNORED_ATTRIBUTES = {("/", "git_commit")}
NORMALIZED_DATASETS = {
    "meta/resolved_input_json": (("output", "directory"),),
    "meta/input_segments_json": (
        ("git_commit",),
        ("resolved_input", "output", "directory"),
    ),
}


@dataclass(frozen=True)
class NumericDifference:
    path: str
    exact: bool
    max_absolute: float
    max_relative: float


@dataclass(frozen=True)
class ComparisonResult:
    reference: str
    candidate: str
    passed: bool
    exact_numeric_datasets: int
    tolerance_numeric_datasets: int
    worst_absolute: NumericDifference | None
    worst_relative: NumericDifference | None
    structural_errors: tuple[str, ...]
    normalized_datasets: tuple[str, ...] = tuple(
        f"{path}:{','.join('.'.join(keys) for keys in key_paths)}"
        for path, key_paths in sorted(NORMALIZED_DATASETS.items())
    )
    ignored_attributes: tuple[str, ...] = tuple(
        f"{path}:{name}" for path, name in sorted(IGNORED_ATTRIBUTES)
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _object_paths(archive: h5py.File) -> dict[str, str]:
    paths = {"/": "group"}

    def record(name: str, item: h5py.Group | h5py.Dataset) -> None:
        paths[name] = "dataset" if isinstance(item, h5py.Dataset) else "group"

    archive.visititems(record)
    return paths


def _attribute_equal(reference: Any, candidate: Any) -> bool:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    return (
        reference_array.shape == candidate_array.shape
        and reference_array.dtype.kind == candidate_array.dtype.kind
        and np.array_equal(reference_array, candidate_array)
    )


def _compare_attributes(
    path: str,
    reference: h5py.Group | h5py.Dataset,
    candidate: h5py.Group | h5py.Dataset,
    errors: list[str],
) -> None:
    reference_names = set(reference.attrs)
    candidate_names = set(candidate.attrs)
    ignored = {name for object_path, name in IGNORED_ATTRIBUTES if object_path == path}
    reference_names -= ignored
    candidate_names -= ignored
    if reference_names != candidate_names:
        errors.append(
            f"{path}: attribute names differ: reference={sorted(reference_names)}, "
            f"candidate={sorted(candidate_names)}"
        )
        return
    for name in sorted(reference_names):
        if not _attribute_equal(reference.attrs[name], candidate.attrs[name]):
            errors.append(f"{path}: attribute {name!r} differs")


def _numeric_difference(
    path: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> NumericDifference:
    if reference.size == 0:
        return NumericDifference(path, True, 0.0, 0.0)
    exact = bool(np.array_equal(reference, candidate))
    absolute = np.abs(candidate - reference)
    scale = np.maximum(np.abs(reference), np.abs(candidate))
    relative = np.divide(absolute, scale, out=np.zeros_like(absolute, dtype=float), where=scale > 0.0)
    return NumericDifference(
        path,
        exact,
        float(np.max(absolute)),
        float(np.max(relative)),
    )


def _normalized_json(
    dataset: h5py.Dataset, key_paths: tuple[tuple[str, ...], ...]
) -> Any:
    raw = dataset[()]
    values = raw.tolist() if isinstance(raw, np.ndarray) else [raw]
    parsed_values = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = json.loads(str(value))
        for keys in key_paths:
            parent = parsed
            for key in keys[:-1]:
                parent = parent[key]
            parent.pop(keys[-1], None)
        parsed_values.append(parsed)
    parsed = parsed_values if isinstance(raw, np.ndarray) else parsed_values[0]
    return parsed


def compare_databases(
    reference_path: Path,
    candidate_path: Path,
    *,
    rtol: float = 5.0e-13,
    atol: float = 5.0e-13,
) -> ComparisonResult:
    errors: list[str] = []
    differences: list[NumericDifference] = []
    tolerance_numeric_datasets = 0
    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        reference_objects = _object_paths(reference)
        candidate_objects = _object_paths(candidate)
        if reference_objects != candidate_objects:
            missing = sorted(set(reference_objects) - set(candidate_objects))
            extra = sorted(set(candidate_objects) - set(reference_objects))
            changed = sorted(
                path
                for path in set(reference_objects) & set(candidate_objects)
                if reference_objects[path] != candidate_objects[path]
            )
            if missing:
                errors.append(f"missing candidate objects: {missing}")
            if extra:
                errors.append(f"extra candidate objects: {extra}")
            if changed:
                errors.append(f"object types differ: {changed}")

        for path in sorted(set(reference_objects) & set(candidate_objects)):
            reference_object = reference if path == "/" else reference[path]
            candidate_object = candidate if path == "/" else candidate[path]
            if type(reference_object) is not type(candidate_object):
                continue
            _compare_attributes(path, reference_object, candidate_object, errors)
            if not isinstance(reference_object, h5py.Dataset):
                continue
            if path in NORMALIZED_DATASETS:
                keys = NORMALIZED_DATASETS[path]
                if _normalized_json(reference_object, keys) != _normalized_json(
                    candidate_object, keys
                ):
                    errors.append(
                        f"{path}: normalized JSON differs after removing {'.'.join(keys)}"
                    )
                continue
            if reference_object.shape != candidate_object.shape:
                errors.append(
                    f"{path}: shapes differ: {reference_object.shape} != {candidate_object.shape}"
                )
                continue
            if reference_object.dtype != candidate_object.dtype:
                errors.append(
                    f"{path}: dtypes differ: {reference_object.dtype} != {candidate_object.dtype}"
                )
                continue
            reference_values = reference_object[...]
            candidate_values = candidate_object[...]
            if np.issubdtype(reference_object.dtype, np.number):
                difference = _numeric_difference(path, reference_values, candidate_values)
                differences.append(difference)
                if not np.allclose(
                    candidate_values,
                    reference_values,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                ):
                    errors.append(
                        f"{path}: numeric values exceed tolerance "
                        f"(max_abs={difference.max_absolute:.6e}, "
                        f"max_rel={difference.max_relative:.6e})"
                    )
                elif not difference.exact:
                    tolerance_numeric_datasets += 1
            elif not np.array_equal(reference_values, candidate_values):
                errors.append(f"{path}: nonnumeric values differ")

    worst_absolute = max(differences, key=lambda item: item.max_absolute, default=None)
    worst_relative = max(differences, key=lambda item: item.max_relative, default=None)
    exact = sum(difference.exact for difference in differences)
    return ComparisonResult(
        str(reference_path),
        str(candidate_path),
        not errors,
        exact,
        tolerance_numeric_datasets,
        worst_absolute,
        worst_relative,
        tuple(errors),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="compare two FE result databases while ignoring run-location provenance"
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=5.0e-13)
    parser.add_argument("--atol", type=float, default=5.0e-13)
    parser.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args(argv)
    result = compare_databases(args.reference, args.candidate, rtol=args.rtol, atol=args.atol)
    report = result.as_dict()
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
