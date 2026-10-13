"""MPI linear-solve service; rank zero retains all FE and output ownership.

Importing this module does not initialize MPI. Only an explicit PetscSession
loads petsc4py/mpi4py; element workers must never enter that session.
"""
from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from .types import ModelError, RecoverableError


DEFAULT_OPTIONS = Path(__file__).with_name("petsc_lgmres.toml")


def load_petsc_options(path: str | Path | None = None) -> dict:
    """Read recommended defaults plus an optional flat TOML override file."""
    settings = {}
    paths = [DEFAULT_OPTIONS] if path is None else [DEFAULT_OPTIONS, Path(path)]
    for source in paths:
        try:
            with source.open("rb") as stream:
                values = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ModelError(f"cannot read PETSc options file {source}: {exc}") from exc
        for key, value in values.items():
            if not key or not key.replace("_", "").isalnum() or not isinstance(
                value, (str, int, float, bool)
            ) or isinstance(value, float) and not np.isfinite(value):
                raise ModelError("PETSc options must be a flat table of finite scalar values")
        settings.update(values)
    return settings


def partition(n: int, size: int, rank: int) -> tuple[int, int]:
    base, extra = divmod(n, size)
    start = rank * base + min(rank, extra)
    return start, start + base + (rank < extra)


class PetscSession:
    """One persistent MPI team, with one active KKT context at a time.

    Rank zero issues setup/solve/close commands. Other ranks wait in serve().
    The context follows the existing exact/modified-Newton factor lifecycle;
    closing it does not shut down the persistent team.
    """

    def __init__(self, options_file: str | Path | None = None):
        try:
            from mpi4py import MPI
            from petsc4py import PETSc
        except (ImportError, OSError) as exc:
            raise ModelError(
                "petsc_lgmres requires compatible PETSc, petsc4py and mpi4py installations"
            ) from exc
        self.MPI, self.PETSc = MPI, PETSc
        self.comm = MPI.COMM_WORLD
        self.rank, self.size = self.comm.rank, self.comm.size
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise ModelError("petsc_lgmres requires a real float64 PETSc build")
        self.options_file = options_file
        self.ksp = None
        self.matrix = self.rhs = self.solution = self.work = None
        self.index_sets = []
        self.diagnostics = {}
        self._root_matrix = None
        self._thread_limit = None

    def __enter__(self):
        # Communicate input errors before non-root ranks enter their command loop.
        error, settings = None, None
        if self.rank == 0:
            try:
                settings = load_petsc_options(self.options_file)
            except ModelError as exc:
                error = str(exc)
        error, self.settings = self.comm.bcast((error, settings), root=0)
        if error is not None:
            raise ModelError(error)
        self._thread_limit = threadpool_limits(limits=1)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.rank == 0:
                self.close()
                self.comm.bcast(("stop", 2 if exc_type else 0), root=0)
        finally:
            if self._thread_limit is not None:
                self._thread_limit.restore_original_limits()

    @property
    def info(self) -> dict:
        return {
            "mpi_ranks": self.size,
            "threads_per_rank": 1,
            "petsc_version": list(self.PETSc.Sys.getVersion()),
            "petsc_options": dict(self.settings),
        }

    def serve(self) -> int:
        """Non-root ranks execute only collective linear-algebra operations."""
        if self.rank == 0:
            raise ModelError("rank zero must run the FE analysis, not the PETSc service loop")
        while True:
            command = self.comm.bcast(None, root=0)
            if command[0] == "stop":
                return command[1]
            if command[0] == "setup":
                self._collective(self._setup, command[1], command[2], None)
            elif command[0] == "solve":
                self._collective(self._solve, None)
            elif command[0] == "close":
                self._collective(self._close)
            else:
                self.comm.Abort(2)

    def _collective(self, operation, *args):
        try:
            return operation(*args)
        except BaseException as exc:
            # A rank-local exception inside a collective can strand its peers.
            # Aborting is reserved for unexpected collective/setup errors, not
            # ordinary KSP nonconvergence (which completes and requests cutback).
            if self.size > 1:
                print(f"PETSc collective failure on rank {self.rank}: {exc}", flush=True)
                self.comm.Abort(2)
            raise ModelError(f"PETSc setup/collective failure: {exc}") from exc

    def setup(self, kkt, displacement_dofs: int):
        if self.rank != 0 or self.ksp is not None:
            raise ModelError("PETSc setup requires rank zero and a closed previous context")
        kkt = kkt.tocsr()
        n = kkt.shape[0]
        chunks = [
            kkt[slice(*partition(n, self.size, rank)), :].tocsr()
            for rank in range(self.size)
        ]
        self._root_matrix = kkt
        self.comm.bcast(("setup", displacement_dofs, n), root=0)
        self._collective(self._setup, displacement_dofs, n, chunks)
        return self

    def _setup(self, displacement_dofs: int, n: int, chunks):
        PETSc = self.PETSc
        local = self.comm.scatter(chunks, root=0)
        start, end = partition(n, self.size, self.rank)
        self.displacement_dofs, self.n = displacement_dofs, n
        self.matrix = PETSc.Mat().createAIJ(
            size=((end - start, n), (PETSc.DECIDE, n)),
            csr=(np.asarray(local.indptr, dtype=PETSc.IntType),
                 np.asarray(local.indices, dtype=PETSc.IntType),
                 np.asarray(local.data, dtype=PETSc.ScalarType)),
            comm=PETSc.COMM_WORLD,
        )
        self.matrix.assemble()
        if self.matrix.getOwnershipRange() != (start, end):
            raise ModelError("PETSc matrix ownership differs from the requested row partition")
        self.rhs = PETSc.Vec().createMPI((end - start, n), comm=PETSc.COMM_WORLD)
        self.solution, self.work = self.rhs.duplicate(), self.rhs.duplicate()
        self.ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        self.ksp.setOptionsPrefix("fe_")
        self.ksp.setOperators(self.matrix)
        for offset, count in ((0, displacement_dofs), (displacement_dofs, n - displacement_dofs)):
            lo, hi = partition(count, self.size, self.rank)
            self.index_sets.append(PETSc.IS().createStride(
                hi - lo, first=offset + lo, step=1, comm=PETSc.COMM_WORLD,
            ))
        pc = self.ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        pc.setFieldSplitIS(("u", self.index_sets[0]), ("lambda", self.index_sets[1]))

        # The options file is authoritative. Restore PETSc's global options
        # database after setup so successive analyses do not inherit our knobs.
        options = PETSc.Options()
        saved = {}
        try:
            for key, value in self.settings.items():
                name = "fe_" + key
                saved[name] = options.getString(name) if options.hasName(name) else None
                options[name] = value
            self.ksp.setFromOptions()
            if (self.ksp.getType() != "lgmres"
                    or self.ksp.getPCSide() != PETSc.PC.Side.RIGHT
                    or self.ksp.getNormType() != PETSc.KSP.NormType.UNPRECONDITIONED
                    or pc.getType() != "fieldsplit"):
                raise ModelError("petsc_lgmres requires LGMRES, RIGHT/UNPRECONDITIONED and FIELDSPLIT")
            self.ksp.setUp()
            self.ksp.setUpOnBlocks()
        finally:
            for name, previous in saved.items():
                if previous is None:
                    del options[name]
                else:
                    options[name] = previous
        self.rtol, self.atol, _, self.max_iterations = self.ksp.getTolerances()
        self.counts = np.array([
            partition(n, self.size, rank)[1] - partition(n, self.size, rank)[0]
            for rank in range(self.size)
        ], dtype=np.int32)
        self.displacements = np.array([
            partition(n, self.size, rank)[0] for rank in range(self.size)
        ], dtype=np.int32)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        if rhs.shape != (self.n,) or not np.all(np.isfinite(rhs)):
            raise ModelError("invalid PETSc KKT right-hand side")
        self.comm.bcast(("solve",), root=0)
        correction = self._collective(self._solve, rhs)
        residual = self._root_matrix @ correction - rhs
        self.diagnostics.update({
            "force_linear_residual_inf": float(np.max(np.abs(residual[:self.displacement_dofs]), initial=0.0)),
            "constraint_linear_residual_inf": float(np.max(np.abs(residual[self.displacement_dofs:]), initial=0.0)),
        })
        if (self.diagnostics["converged_reason"] <= 0
                or not np.all(np.isfinite(correction))
                or not np.isfinite(self.diagnostics["true_residual_l2"])
                or self.diagnostics["true_residual_l2"] > self.diagnostics["residual_tolerance"]):
            raise RecoverableError(
                "PETSc KKT solve failed accuracy/convergence check: "
                f"reason={self.diagnostics['converged_reason']}, "
                f"true relative residual={self.diagnostics['true_relative_residual']:.3e}"
            )
        return correction

    def _solve(self, rhs):
        send = [rhs, self.counts, self.displacements, self.MPI.DOUBLE] if self.rank == 0 else None
        self.comm.Scatterv(send, self.rhs.getArray(), root=0)
        self.solution.set(0.0)
        b_norm = float(self.rhs.norm())
        if b_norm:
            self.ksp.solve(self.rhs, self.solution)
            reason = int(self.ksp.getConvergedReason())
            iterations = int(self.ksp.getIterationNumber())
            reported = float(self.ksp.getResidualNorm())
        else:
            reason = int(self.PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
            iterations, reported = 0, 0.0
        self.matrix.mult(self.solution, self.work)
        self.work.axpy(-1.0, self.rhs)
        true_norm = float(self.work.norm())
        self.diagnostics = {
            "iterations": iterations,
            "converged_reason": reason,
            "reported_residual_l2": reported,
            "true_residual_l2": true_norm,
            "true_relative_residual": true_norm / b_norm if b_norm else 0.0,
            "residual_tolerance": max(self.rtol * b_norm, self.atol),
        }
        result = np.empty(self.n) if self.rank == 0 else None
        receive = [result, self.counts, self.displacements, self.MPI.DOUBLE] if self.rank == 0 else None
        self.comm.Gatherv(self.solution.getArray(readonly=True), receive, root=0)
        return result

    def close(self):
        if self.rank == 0 and self.matrix is not None:
            self.comm.bcast(("close",), root=0)
            self._collective(self._close)

    def _close(self):
        if self.ksp is not None:
            self.ksp.destroy()
        for item in self.index_sets:
            item.destroy()
        self.index_sets = []
        for item in (self.work, self.solution, self.rhs, self.matrix):
            if item is not None:
                item.destroy()
        self.ksp = self.matrix = self.rhs = self.solution = self.work = None
        self._root_matrix = None
