"""Python material wrappers for the external multiphase TRIP C ABI."""
from __future__ import annotations

from collections.abc import Mapping
import ctypes
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .types import (
    EvaluationStatus,
    FailureKind,
    MaterialInitRequest,
    MaterialInitResponse,
    MaterialRequest,
    MaterialResponse,
    ModelError,
    StateField,
    StateLayout,
)


CP_OK = 0
CP_INVALID_STATE_SIZE = 1
CP_INVALID_ORIENTATION = 2
CP_INVALID_TIME = 3
CP_INVALID_DEFORMATION = 5
CP_NONFINITE_STATE = 6
CP_NONFINITE_RESPONSE = 7

_ABI_VERSION = 4
_PHASES = {
    "bcc": (1, 93),
    "fcc": (2, 151),
}
_STATUS_MESSAGES = {
    CP_INVALID_STATE_SIZE: "invalid multiphase TRIP state size",
    CP_INVALID_ORIENTATION: "multiphase TRIP Euler angles must be finite",
    CP_INVALID_TIME: "invalid multiphase TRIP increment times",
    CP_INVALID_DEFORMATION: "invalid multiphase TRIP deformation gradient",
    CP_NONFINITE_STATE: "non-finite committed multiphase TRIP state",
    CP_NONFINITE_RESPONSE: "non-finite multiphase TRIP response",
}
_RECOVERABLE_STATUSES = {CP_INVALID_DEFORMATION, CP_NONFINITE_RESPONSE}
_VECTOR = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")

BCC_STATE_LAYOUT = StateLayout(
    (StateField("internal_variables", (93,), tuple(str(i) for i in range(93))),)
)
FCC_STATE_LAYOUT = StateLayout(
    (StateField("internal_variables", (151,), tuple(str(i) for i in range(151))),)
)


class _TRIPLibrary:
    def __init__(self, path: Path):
        try:
            self.handle = ctypes.CDLL(str(path))
        except OSError as exc:
            raise ModelError(f"cannot load multiphase TRIP library {path}: {exc}") from exc
        scalar_names = (
            "cp_abi_version",
            "cp_bcc_state_size",
            "cp_fcc_state_size",
        )
        try:
            for name in scalar_names:
                function = getattr(self.handle, name)
                function.argtypes = []
                function.restype = ctypes.c_int
            for phase in _PHASES:
                initialize = getattr(self.handle, f"cp_{phase}_initialize")
                initialize.argtypes = [ctypes.c_int, _VECTOR]
                initialize.restype = ctypes.c_int
                update = getattr(self.handle, f"cp_{phase}_update")
                update.argtypes = [
                    ctypes.c_int,
                    _VECTOR,
                    ctypes.c_double,
                    ctypes.c_double,
                    _VECTOR,
                    _VECTOR,
                    _VECTOR,
                    _VECTOR,
                    _VECTOR,
                    _VECTOR,
                ]
                update.restype = ctypes.c_int
        except AttributeError as exc:
            raise ModelError(
                f"multiphase TRIP library {path} does not provide ABI version {_ABI_VERSION}"
            ) from exc
        reported_abi = self.handle.cp_abi_version()
        if reported_abi != _ABI_VERSION:
            raise ModelError(
                f"multiphase TRIP library {path} has ABI version {reported_abi}, "
                f"expected {_ABI_VERSION}; rebuild the shared library"
            )
        for phase, (_phase_id, state_size) in _PHASES.items():
            reported = getattr(self.handle, f"cp_{phase}_state_size")()
            if reported != state_size:
                raise ModelError(
                    f"multiphase TRIP {phase} library state size is {reported}, expected {state_size}"
                )


_LIBRARIES: dict[tuple[str, str], _TRIPLibrary] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trip_properties(properties: Mapping[str, object]) -> Mapping[str, object]:
    unknown = set(properties) - {"library"}
    if unknown:
        raise ModelError(f"unknown multiphase TRIP properties: {sorted(unknown)}")
    missing = {"library"} - set(properties)
    if missing:
        raise ModelError(f"missing multiphase TRIP properties: {sorted(missing)}")
    library = str(properties["library"])
    if not library:
        raise ModelError("multiphase TRIP library path must not be empty")
    return MappingProxyType({"library": library})


def resolve_trip_properties(
    properties: Mapping[str, object], base_directory: Path
) -> Mapping[str, object]:
    result = dict(properties)
    declared = Path(str(result["library"])).expanduser()
    resolved = declared if declared.is_absolute() else base_directory / declared
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ModelError(f"multiphase TRIP library does not exist: {resolved}")
    try:
        digest = _sha256(resolved)
    except OSError as exc:
        raise ModelError(f"cannot read multiphase TRIP library {resolved}") from exc
    result["library_sha256"] = digest
    result["_resolved_library"] = str(resolved)
    return MappingProxyType(result)


def _load_library(properties: Mapping[str, object]) -> _TRIPLibrary:
    try:
        path = Path(str(properties["_resolved_library"]))
        expected_digest = str(properties["library_sha256"])
    except KeyError as exc:
        raise ModelError("multiphase TRIP library properties were not resolved") from exc
    key = (str(path), expected_digest)
    # Each process hashes and loads once; subsequent point calls use this cache.
    library = _LIBRARIES.get(key)
    if library is not None:
        return library
    try:
        actual_digest = _sha256(path)
    except OSError as exc:
        raise ModelError(f"cannot read multiphase TRIP library {path}") from exc
    if actual_digest != expected_digest:
        raise ModelError(
            f"multiphase TRIP library changed after model construction: {path}"
        )
    library = _TRIPLibrary(path)
    _LIBRARIES[key] = library
    return library


def _point_orientation(
    point_properties: object | None,
    phase: str,
) -> np.ndarray:
    if not isinstance(point_properties, Mapping):
        raise ModelError("multiphase TRIP requires point properties")
    try:
        raw_phase = np.asarray(point_properties["phase_id"])
        angles = np.asarray(point_properties["euler_angles"], dtype=np.float64)
    except KeyError as exc:
        raise ModelError(
            "multiphase TRIP requires phase_id and euler_angles point properties"
        ) from exc
    if raw_phase.shape:
        raise ModelError("multiphase TRIP phase_id must be scalar")
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ModelError("multiphase TRIP euler_angles must be a finite vector of shape (3,) in radians")
    phase_value = raw_phase.item()
    try:
        phase_id = int(phase_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelError("multiphase TRIP phase_id must be an integer") from exc
    if phase_id != phase_value:
        raise ModelError("multiphase TRIP phase_id must be an integer")
    expected_phase, _state_size = _PHASES[phase]
    if phase_id != expected_phase:
        raise ModelError(
            f"multiphase TRIP {phase} model received phase_id={phase_id}, "
            f"expected {expected_phase}"
        )
    return np.ascontiguousarray(angles)


def _initialize(phase: str, request: MaterialInitRequest, layout: StateLayout) -> MaterialInitResponse:
    try:
        library = _load_library(request.properties)
        # Initialization does not use orientation, but validate the assignment now
        # so an invalid phase/angle vector fails before the first material update.
        _point_orientation(request.point_properties, phase)
        state = np.empty(layout.n_state, dtype=np.float64)
        status = getattr(library.handle, f"cp_{phase}_initialize")(
            layout.n_state, state
        )
        if status != CP_OK:
            message = _STATUS_MESSAGES.get(status, f"unknown ABI status {status}")
            return MaterialInitResponse(
                np.empty(layout.n_state), EvaluationStatus(FailureKind.FATAL, message)
            )
        return MaterialInitResponse(state, EvaluationStatus())
    except (ModelError, TypeError, ValueError) as exc:
        return MaterialInitResponse(
            np.empty(layout.n_state), EvaluationStatus(FailureKind.FATAL, str(exc))
        )


def init_multiphase_trip_ferrite(request: MaterialInitRequest) -> MaterialInitResponse:
    return _initialize("bcc", request, BCC_STATE_LAYOUT)


def init_multiphase_trip_austenite(request: MaterialInitRequest) -> MaterialInitResponse:
    return _initialize("fcc", request, FCC_STATE_LAYOUT)


def _failure(
    request: MaterialRequest, kind: FailureKind, message: str
) -> MaterialResponse:
    return MaterialResponse(
        np.zeros((3, 3)), None, request.state_n, EvaluationStatus(kind, message)
    )


def _update(
    phase: str, request: MaterialRequest, layout: StateLayout
) -> MaterialResponse:
    if request.state_n.layout != layout:
        return _failure(request, FailureKind.FATAL, "invalid multiphase TRIP state layout")
    try:
        library = _load_library(request.properties)
        angles = _point_orientation(request.point_properties, phase)
        F_n_array = np.asarray(request.F_n, dtype=np.float64)
        F_np1_array = np.asarray(request.F_np1, dtype=np.float64)
        if F_n_array.shape != (3, 3) or F_np1_array.shape != (3, 3):
            raise ModelError("multiphase TRIP deformation gradients must be 3-by-3")
        F_n = np.ascontiguousarray(F_n_array).reshape(-1)
        F_np1 = np.ascontiguousarray(F_np1_array).reshape(-1)
        state_n = np.ascontiguousarray(request.state_n.values, dtype=np.float64)
        P = np.empty(9, dtype=np.float64)
        A = np.empty(81, dtype=np.float64)
        state_trial = np.empty(layout.n_state, dtype=np.float64)
        # UMAT always computes dP/dF: need_tangent stops at this wrapper boundary.
        # It only controls whether the common API returns that computed tangent.
        status = getattr(library.handle, f"cp_{phase}_update")(
            layout.n_state,
            angles,
            float(request.t_n),
            float(request.t_np1),
            F_n,
            F_np1,
            state_n,
            P,
            A,
            state_trial,
        )
    except (ModelError, TypeError, ValueError) as exc:
        return _failure(request, FailureKind.FATAL, str(exc))
    if status != CP_OK:
        kind = (
            FailureKind.RECOVERABLE
            if status in _RECOVERABLE_STATUSES
            else FailureKind.FATAL
        )
        return _failure(
            request, kind, _STATUS_MESSAGES.get(status, f"unknown ABI status {status}")
        )
    tangent = A.reshape(3, 3, 3, 3) if request.need_tangent else None
    return MaterialResponse(
        P.reshape(3, 3), tangent, layout.view(state_trial), EvaluationStatus()
    )


def update_multiphase_trip_ferrite(request: MaterialRequest) -> MaterialResponse:
    return _update("bcc", request, BCC_STATE_LAYOUT)


def update_multiphase_trip_austenite(request: MaterialRequest) -> MaterialResponse:
    return _update("fcc", request, FCC_STATE_LAYOUT)
