"""Execution policy for deterministic element-level parallelism."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Iterable, Iterator

import numpy as np

from .elements import evaluate_element
from .j2_kernel import active_j2_backend, available_j2_backends, configure_j2_backend
from .types import ElementRequest, ElementResponse, MaterialDefinition, MaterialModel, ModelError


@dataclass(frozen=True)
class ElementWorkItem:
    """Serializable element-local input identified by its model block."""

    block_index: int
    element_index: int
    X_e: np.ndarray
    u_e_n: np.ndarray
    u_e_trial: np.ndarray
    state_e_n: np.ndarray
    point_properties: object | None
    t_n: float
    t_np1: float
    need_tangent: bool
    formulation: str

    def request(self, material: MaterialDefinition) -> ElementRequest:
        return ElementRequest(
            self.X_e,
            self.u_e_n,
            self.u_e_trial,
            self.state_e_n,
            material,
            self.point_properties,
            self.t_n,
            self.t_np1,
            self.need_tangent,
            self.formulation,
        )


@dataclass(frozen=True)
class _SerializableMaterial:
    name: str
    model: MaterialModel
    properties: dict[str, object]


@dataclass(frozen=True)
class ExecutionInfo:
    element_backend: str
    j2_backend: str
    requested_processes: int
    effective_processes: int
    available_physical_cores: int | None
    available_logical_cpus: int
    process_start_method: str | None
    worker_blas_threads: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_WORKER_MATERIALS: tuple[MaterialDefinition, ...] = ()
_WORKER_THREAD_LIMITER: object | None = None


def _initialize_worker(
    materials: tuple[_SerializableMaterial, ...], j2_backend: str
) -> None:
    """Rebuild immutable material definitions and prevent nested BLAS pools."""
    global _WORKER_MATERIALS, _WORKER_THREAD_LIMITER
    from threadpoolctl import threadpool_limits

    _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    configure_j2_backend(j2_backend)
    _WORKER_MATERIALS = tuple(
        MaterialDefinition(
            material.name,
            material.model,
            MappingProxyType(dict(material.properties)),
        )
        for material in materials
    )


def _evaluate_worker_item(item: ElementWorkItem) -> ElementResponse:
    return evaluate_element(item.request(_WORKER_MATERIALS[item.block_index]))


def _available_cpu_ids() -> set[int]:
    try:
        return set(os.sched_getaffinity(0))
    except AttributeError:
        count = os.process_cpu_count() or os.cpu_count() or 1
        return set(range(count))


def _available_physical_cores(cpu_ids: set[int]) -> int | None:
    topology: set[tuple[str, str]] = set()
    try:
        for cpu in cpu_ids:
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            package = (root / "physical_package_id").read_text(encoding="ascii").strip()
            core = (root / "core_id").read_text(encoding="ascii").strip()
            topology.add((package, core))
    except OSError:
        return None
    return len(topology) or None


def _process_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    return mp.get_context("forkserver" if "forkserver" in methods else "spawn")


class ElementExecutor:
    """Persistent serial or process executor for complete element kernels."""

    def __init__(
        self,
        materials: Iterable[MaterialDefinition],
        element_count: int,
        num_processes: int,
        *,
        j2_backend: str = "python",
    ):
        if (
            isinstance(num_processes, bool)
            or not isinstance(num_processes, (int, np.integer))
            or num_processes < 1
        ):
            raise ModelError("--num-processes must be a positive integer")
        if element_count < 1:
            raise ModelError("cannot create an element executor for an empty model")
        if j2_backend not in available_j2_backends():
            available = ", ".join(available_j2_backends())
            raise ModelError(
                f"J2 backend {j2_backend!r} is unavailable; available backends: {available}"
            )
        self._materials = tuple(materials)
        if not self._materials:
            raise ModelError("cannot create an element executor without material blocks")
        requested = int(num_processes)
        effective = min(requested, element_count)
        cpu_ids = _available_cpu_ids()
        self._pool: ProcessPoolExecutor | None = None
        self._parallel = effective > 1
        start_method: str | None = None
        worker_blas_threads: int | None = None
        serializable: tuple[_SerializableMaterial, ...] = ()
        if self._parallel:
            serializable = tuple(
                _SerializableMaterial(material.name, material.model, dict(material.properties))
                for material in self._materials
            )
            try:
                pickle.dumps(serializable)
            except (pickle.PickleError, TypeError, AttributeError) as exc:
                raise ModelError(
                    "process element assembly requires importable material routines and picklable properties"
                ) from exc
        self._previous_j2_backend = active_j2_backend()
        configure_j2_backend(j2_backend)
        try:
            if self._parallel:
                context = _process_context()
                start_method = context.get_start_method()
                worker_blas_threads = 1
                self._pool = ProcessPoolExecutor(
                    max_workers=effective,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(serializable, j2_backend),
                )
        except Exception:
            configure_j2_backend(self._previous_j2_backend)
            raise
        self.info = ExecutionInfo(
            "process" if self._parallel else "serial",
            j2_backend,
            requested,
            effective,
            _available_physical_cores(cpu_ids),
            len(cpu_ids),
            start_method,
            worker_blas_threads,
        )

    @property
    def parallel(self) -> bool:
        return self._parallel

    def describe(self) -> str:
        physical = (
            str(self.info.available_physical_cores)
            if self.info.available_physical_cores is not None
            else "unknown"
        )
        return (
            f"element assembly: {self.info.element_backend}, "
            f"{self.info.requested_processes} process(es) requested, "
            f"{self.info.effective_processes} effective; "
            f"{physical} physical core(s) and "
            f"{self.info.available_logical_cpus} logical CPU(s) available; "
            f"J2 backend: {self.info.j2_backend}"
        )

    def evaluate(self, items: list[ElementWorkItem]) -> Iterator[ElementResponse]:
        if not self._parallel:
            for item in items:
                yield evaluate_element(item.request(self._materials[item.block_index]))
            return
        assert self._pool is not None
        chunksize = max(1, math.ceil(len(items) / (4 * self.info.effective_processes)))
        try:
            yield from self._pool.map(_evaluate_worker_item, items, chunksize=chunksize)
        except Exception as exc:
            raise ModelError(f"parallel element evaluation failed: {exc}") from exc

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        configure_j2_backend(self._previous_j2_backend)

    def __enter__(self) -> "ElementExecutor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
