from __future__ import annotations

import os
import sys

# Keep each MPI rank single-threaded so MPI scaling is not obscured by
# BLAS/OpenMP oversubscription.
_BENCH_NUM_THREADS = os.environ.get("BENCH_NUM_THREADS", "1")
_BENCH_MODE = os.environ.get("BENCH_MODE", "validate").strip().lower()
_BENCH_TARGET_ITS_RAW = os.environ.get("BENCH_TARGET_ITS")

if _BENCH_MODE not in {"validate", "timing"}:
    raise ValueError("BENCH_MODE must be 'validate' or 'timing'")

if _BENCH_MODE == "timing" and _BENCH_TARGET_ITS_RAW is None:
    raise ValueError(
        "BENCH_MODE=timing requires BENCH_TARGET_ITS=<validated iteration count>"
    )

_BENCH_TARGET_ITS = (
    int(_BENCH_TARGET_ITS_RAW)
    if _BENCH_TARGET_ITS_RAW is not None
    else None
)

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
):
    os.environ[_name] = _BENCH_NUM_THREADS


# PETSc must see command-line arguments before PETSc is imported.
import petsc4py

petsc4py.init(sys.argv)

from pathlib import Path
from time import perf_counter

import h5py
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy import sparse


def root_print(
    comm: MPI.Comm,
    *args: object,
    **kwargs: object,
) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)


def partition_1d(
    n: int,
    size: int,
    rank: int,
) -> tuple[int, int]:
    """
    Contiguous, near-even partition [start, end) of range(n).
    """
    base, rem = divmod(n, size)

    local_n = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)

    return start, start + local_n


def partition_counts(
    n: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return counts and displacements for partition_1d().
    """
    counts = np.empty(size, dtype=np.int32)
    displs = np.empty(size, dtype=np.int32)

    for rank in range(size):
        start, end = partition_1d(n, size, rank)
        counts[rank] = end - start
        displs[rank] = start

    return counts, displs


def mpi_dtype_for_numpy(
    dtype: np.dtype,
) -> MPI.Datatype:
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.float64):
        return MPI.DOUBLE

    if dtype == np.dtype(np.float32):
        return MPI.FLOAT

    raise TypeError(
        "This benchmark currently supports real float32/float64 "
        f"PETSc scalars, got {dtype}"
    )


def _read_csr(
    group: h5py.Group,
    name: str,
) -> sparse.csr_matrix:
    g = group[name]

    shape = tuple(
        int(value)
        for value in g.attrs["shape"]
    )

    return sparse.csr_matrix(
        (
            np.asarray(g["data"]),
            np.asarray(g["indices"]),
            np.asarray(g["indptr"]),
        ),
        shape=shape,
        copy=False,
    )


def load_problem(
    path: Path,
    group_name: str,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    sparse.csr_matrix,
    np.ndarray,
]:
    """
    Load a replicated SciPy copy of the KKT problem on every MPI rank.

    The solve itself is distributed in PETSc.  Replicated input loading is
    retained here to keep the benchmark/data format simple and comparable
    with the previous implementation.
    """
    with h5py.File(path, "r") as h5:
        group = h5[group_name]

        K = _read_csr(group, "K")
        C = _read_csr(group, "C")

        r_u = np.asarray(group["r_u"])
        r_c = np.asarray(group["r_c"])

    zero = sparse.csr_matrix(
        (C.shape[0], C.shape[0]),
        dtype=K.dtype,
    )

    A = sparse.bmat(
        [
            [K, C.T],
            [C, zero],
        ],
        format="csr",
    )

    b = -np.concatenate(
        (
            r_u,
            r_c,
        )
    )

    return K, C, A, b


def create_distributed_petsc_matrix(
    A: sparse.csr_matrix,
    comm: MPI.Comm,
) -> PETSc.Mat:
    """
    Create an MPIAIJ PETSc matrix from replicated SciPy CSR data.

    Every rank inserts only its owned contiguous row range.
    """
    nrows, ncols = A.shape
    rstart, rend = partition_1d(nrows, comm.size, comm.rank)
    local_rows = rend - rstart
    A_local = A[rstart:rend, :].tocsr()
    indptr = np.asarray(A_local.indptr, dtype=PETSc.IntType)
    indices = np.asarray(A_local.indices, dtype=PETSc.IntType)
    data = np.asarray(A_local.data, dtype=PETSc.ScalarType)
    A_petsc = PETSc.Mat().createAIJ(
        size=((local_rows, nrows), (PETSc.DECIDE, ncols)),
        csr=(indptr, indices, data),
        comm=PETSc.COMM_WORLD,
    )
    A_petsc.assemble()
    actual_start, actual_end = A_petsc.getOwnershipRange()

    if (actual_start, actual_end) != (rstart, rend):
        raise RuntimeError(
            "PETSc row ownership does not match the requested "
            f"partition: expected {(rstart, rend)}, "
            f"got {(actual_start, actual_end)}"
        )

    return A_petsc


def create_distributed_vector(
    values: np.ndarray,
    comm: MPI.Comm,
) -> PETSc.Vec:
    """
    Create a distributed PETSc vector from a replicated NumPy vector.
    """
    n = int(values.size)

    start, end = partition_1d(n, comm.size, comm.rank)

    local_n = end - start

    vec = PETSc.Vec().createMPI((local_n, n), comm=PETSc.COMM_WORLD)

    local = vec.getArray()

    local[:] = np.asarray(values[start:end], dtype=PETSc.ScalarType)

    actual_start, actual_end = vec.getOwnershipRange()

    if (actual_start, actual_end) != (start, end):
        raise RuntimeError(
            "PETSc vector ownership does not match the requested "
            f"partition: expected {(start, end)}, "
            f"got {(actual_start, actual_end)}"
        )

    return vec


def configure_distributed_block_pc(
    ksp: PETSc.KSP,
    n: int,
    m: int,
    comm: MPI.Comm,
) -> tuple[PETSc.IS, PETSc.IS]:
    """
    Configure a fully distributed Schur-complement block
    preconditioner for

        [ K   C^T ]
        [ C    0  ]

    Variable ordering is assumed to be

        [u_0, ..., u_(n-1), lambda_0, ..., lambda_(m-1)]

    The previous implementation gathered the complete residual onto
    MPI rank 0 and applied the SciPy block preconditioner there.

    This implementation keeps the PC application inside PETSc and
    distributed across MPI ranks.

    The Schur factorization is LOWER, corresponding structurally to

        z_u      ~= K^{-1} r_u
        z_lambda ~= S^{-1}(C z_u - r_lambda)

    with PETSc providing distributed approximations to the block
    inverses.
    """

    rank = comm.rank
    size = comm.size

    # --------------------------------------------------------------
    # Distributed index set for u.
    # --------------------------------------------------------------
    u_start, u_end = partition_1d(n, size, rank,)

    is_u = PETSc.IS().createStride(u_end - u_start, first=u_start, step=1, comm=PETSc.COMM_WORLD,)

    # --------------------------------------------------------------
    # Distributed index set for lambda.
    # --------------------------------------------------------------
    lam_start, lam_end = partition_1d(m, size, rank,)

    is_lam = PETSc.IS().createStride(
        lam_end - lam_start,
        first=n + lam_start,
        step=1,
        comm=PETSc.COMM_WORLD,
    )

    # --------------------------------------------------------------
    # PETSc field-split PC.
    # --------------------------------------------------------------
    pc = ksp.getPC()

    pc.setType(PETSc.PC.Type.FIELDSPLIT)

    pc.setFieldSplitIS(("u", is_u), ("lambda", is_lam),)

    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)

    # Lower block factorization:
    #
    #     u solve
    #        |
    #        v
    #     Schur solve
    #
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.LOWER)

    # The actual A11 block is zero for this saddle-point problem.
    #
    # SELFP constructs an explicitly preconditionable approximation
    # based on
    #
    #     S_p ~= -C diag(K)^(-1) C^T
    #
    # instead of attempting to precondition using A11 = 0.
    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.SELFP)

    return is_u, is_lam


def set_petsc_option_default(
    options: PETSc.Options,
    name: str,
    value: str | int | float | bool,
) -> None:
    """
    Set a PETSc option only if the user did not already provide it on
    the command line.
    """
    if not options.hasName(name):
        options[name] = value


def timed_collective(
    comm: MPI.Comm,
    label: str,
    function,
) -> object:
    """
    Time a collective operation and print the maximum rank time.
    """
    comm.Barrier()
    start = perf_counter()
    result = function()
    comm.Barrier()
    local_seconds = perf_counter() - start
    max_seconds = comm.reduce(local_seconds, op=MPI.MAX, root=0)

    if comm.rank == 0:
        assert max_seconds is not None
        print(f"{label}: {max_seconds:.2f} s")

    return result


def distributed_true_relative_residual(
    A: PETSc.Mat,
    x: PETSc.Vec,
    b: PETSc.Vec,
    work: PETSc.Vec,
    b_norm: float,
) -> float:
    """
    Compute

        ||A x - b|| / ||b||

    using distributed PETSc operations.
    """
    A.mult(x, work)

    work.axpy(
        -1.0,
        b,
    )

    return float(
        work.norm() / b_norm
    )


def gather_vector_to_root(
    vec: PETSc.Vec,
    comm: MPI.Comm,
) -> np.ndarray | None:
    """
    Gather a distributed PETSc vector onto rank 0.

    This is used only for the final independent SciPy residual
    cross-check.  It is NOT part of the LGMRES/preconditioner
    iteration.
    """
    counts, displs = partition_counts(vec.getSize(), comm.size)
    local = np.asarray(vec.getArray(readonly=True))
    mpi_dtype = mpi_dtype_for_numpy(local.dtype)
    if comm.rank == 0:
        full = np.empty(vec.getSize(), dtype=local.dtype)
        recvbuf = [full, counts, displs, mpi_dtype]
    else:
        full = None
        recvbuf = None
    comm.Gatherv(local, recvbuf, root=0)

    return full


def main() -> None:
    comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size

    path = Path(__file__).with_name("kkt_medium.h5")
    group_name = "t_n=0.02/iteration=1"

    # --------------------------------------------------------------
    # 1. Load problem.
    # --------------------------------------------------------------
    #
    # The input data remains replicated in this benchmark.
    #
    # The expensive iterative solve itself, including the
    # preconditioner, is distributed.
    #
    K, C, A, b = load_problem(path, group_name)

    n = K.shape[0]
    m = C.shape[0]
    N = A.shape[0]

    if N != n + m:
        raise RuntimeError(f"Unexpected KKT dimensions:  N={N}, n={n}, m={m}")

    root_print(comm, f"MPI ranks: {size}",)
    root_print(comm, f"K: {K.shape}, nnz={K.nnz}",)
    root_print(comm, f"C: {C.shape}, nnz={C.nnz}",)
    root_print(comm, f"A: {A.shape}, nnz={A.nnz}",)
    root_print(comm, f"PETSc version: {PETSc.Sys.getVersion()}",)
    root_print(comm, f"PETSc scalar dtype: {np.dtype(PETSc.ScalarType)}",)
    root_print(comm, f"PETSc integer dtype: {np.dtype(PETSc.IntType)}",)
    root_print(comm, f"threads per MPI rank requested: {_BENCH_NUM_THREADS}",)
    root_print(comm, f"benchmark mode: {_BENCH_MODE}",)
    root_print(comm, "preconditioner: distributed PETSc PCFIELDSPLIT/SCHUR")

    # --------------------------------------------------------------
    # 2. Create distributed PETSc KKT matrix, RHS and solution.
    # --------------------------------------------------------------
    def create_petsc_objects():
        A_petsc = create_distributed_petsc_matrix(A, comm)
        b_petsc = create_distributed_vector(np.asarray(b, dtype=PETSc.ScalarType), comm)
        x_start, x_end = partition_1d(N, size, rank)
        x_petsc = PETSc.Vec().createMPI((x_end - x_start, N), comm=PETSc.COMM_WORLD)
        x_petsc.set(0.0)

        return (A_petsc, b_petsc, x_petsc)

    (
        A_petsc,
        b_petsc,
        x_petsc,
    ) = timed_collective(comm, "distributed PETSc matrix/vector creation", create_petsc_objects)

    # --------------------------------------------------------------
    # 3. Distributed PETSc LGMRES.
    # --------------------------------------------------------------
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOptionsPrefix("bench_")
    ksp.setOperators(A_petsc)
    ksp.setType(PETSc.KSP.Type.LGMRES)
    ksp.setGMRESRestart(33)
    options = PETSc.Options()
    set_petsc_option_default(options, "bench_ksp_lgmres_augment", 3)

    # --------------------------------------------------------------
    # 4. Fully distributed block preconditioner.
    # --------------------------------------------------------------
    #
    # Old implementation:
    #
    #     distributed residual
    #              |
    #              v
    #           Gatherv
    #              |
    #              v
    #          MPI rank 0
    #        SciPy K_ilu.solve
    #        SciPy Schur solve
    #              |
    #              v
    #           Scatterv
    #
    #
    # New implementation:
    #
    #        PETSc PCFIELDSPLIT
    #            /       \
    #           /         \
    #      K block       Schur block
    #        |               |
    #        v               v
    #    distributed     distributed
    #
    is_u, is_lam = configure_distributed_block_pc(ksp, n, m, comm)

    # --------------------------------------------------------------
    # u / K block
    # --------------------------------------------------------------
    #
    # Instead of a single global scipy.spilu(K) solve on rank 0,
    # use additive Schwarz across MPI ranks.
    #
    # Each local Schwarz subproblem uses ILU.
    #
    set_petsc_option_default(options, "bench_fieldsplit_u_ksp_type", "preonly")
    set_petsc_option_default(options, "bench_fieldsplit_u_pc_type", "asm")
    set_petsc_option_default(options, "bench_fieldsplit_u_pc_asm_overlap", 1)
    set_petsc_option_default(options, "bench_fieldsplit_u_sub_ksp_type", "preonly")
    set_petsc_option_default(options, "bench_fieldsplit_u_sub_pc_type", "ilu")
    set_petsc_option_default(options, "bench_fieldsplit_u_sub_pc_factor_levels", 0)
    set_petsc_option_default(options, "bench_fieldsplit_u_pc_asm_type", "basic")


    # --------------------------------------------------------------
    # lambda / Schur block
    # --------------------------------------------------------------
    #
    # With SELFP, PETSc constructs the distributed approximation
    #
    #     -C diag(K)^(-1) C^T
    #
    # Use Jacobi for this Schur approximation.  This is the validated
    # baseline paired with ASM + ILU(1) for the K block.
    #
    set_petsc_option_default(options, "bench_fieldsplit_lambda_ksp_type", "preonly")
    set_petsc_option_default(options, "bench_fieldsplit_lambda_pc_type", "jacobi")

    # --------------------------------------------------------------
    # Outer KSP configuration.
    # --------------------------------------------------------------
    #
    # RIGHT preconditioning preserves the physical residual b - A x.
    # With UNPRECONDITIONED norm, petsc_rnorm is ||b - A x||_2.
    #
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

    rtol = 1e-9
    atol = 0.0
    max_it = 5000

    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    # Command-line PETSc options override the defaults above.
    ksp.setFromOptions()

    # --------------------------------------------------------------
    # 5. Validation / fixed-iteration convergence test.
    # --------------------------------------------------------------
    b_norm = float(b_petsc.norm())

    if b_norm == 0.0:
        raise RuntimeError("RHS norm is zero; relative residual is undefined")

    residual_history: list[tuple[int, float, float]] = []

    if _BENCH_MODE == "validate":
        threshold = max(rtol * b_norm, atol)

        def convergence_test(
            ksp_: PETSc.KSP,
            iteration: int,
            petsc_rnorm: float,
        ) -> PETSc.KSP.ConvergedReason:
            del ksp_

            # With RIGHT preconditioning and UNPRECONDITIONED norm,
            # PETSc already supplies ||b - A x||_2 here.  Recomputing
            # A x inside this callback would add an unnecessary
            # distributed matrix-vector product on every iteration.
            true_abs = float(petsc_rnorm)
            true_rel = true_abs / b_norm

            if rank == 0:
                residual_history.append(
                    (iteration, true_abs, true_rel)
                )

            if not np.isfinite(true_abs):
                return PETSc.KSP.ConvergedReason.DIVERGED_NANORINF

            if true_abs <= threshold:
                if atol > rtol * b_norm:
                    return PETSc.KSP.ConvergedReason.CONVERGED_ATOL

                return PETSc.KSP.ConvergedReason.CONVERGED_RTOL

            if iteration >= max_it:
                return PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT

            return PETSc.KSP.ConvergedReason.ITERATING

    else:
        assert _BENCH_TARGET_ITS is not None

        def convergence_test(
            ksp_: PETSc.KSP,
            iteration: int,
            petsc_rnorm: float,
        ) -> PETSc.KSP.ConvergedReason:
            del ksp_, petsc_rnorm

            if iteration >= _BENCH_TARGET_ITS:
                return PETSc.KSP.ConvergedReason.CONVERGED_ITS

            return PETSc.KSP.ConvergedReason.ITERATING

    ksp.setConvergenceTest(convergence_test)

    # --------------------------------------------------------------
    # 6. PETSc setup.
    # --------------------------------------------------------------
    def setup_ksp_and_blocks() -> None:
        ksp.setUp()
        ksp.setUpOnBlocks()

    timed_collective(comm, "PETSc distributed KSP/PC setup", setup_ksp_and_blocks)

    # Print the actual configured split solvers.
    if rank == 0:
        print(f"KSP type: {ksp.getType()}")
        print(f"PC type: {ksp.getPC().getType()}")
    # --------------------------------------------------------------
    # 7. Solve.
    # --------------------------------------------------------------
    timed_collective(
        comm,
        "PETSc distributed LGMRES solution",
        lambda: ksp.solve(
            b_petsc,
            x_petsc,
        )
    )

    # --------------------------------------------------------------
    # 8. Independent distributed true residual.
    # --------------------------------------------------------------
    # This explicit A x residual is computed once after the solve.
    #
    true_residual_work = b_petsc.duplicate()
    true_rel_petsc = (
        distributed_true_relative_residual(
            A_petsc,
            x_petsc,
            b_petsc,
            true_residual_work,
            b_norm,
        )
    )

    # --------------------------------------------------------------
    # 9. Independent final SciPy residual check on rank 0.
    #
    # This gather occurs ONCE after the solve.
    #
    # It is not part of the LGMRES iteration and therefore does not
    # reintroduce the old preconditioner bottleneck.
    # --------------------------------------------------------------
    x_global = gather_vector_to_root(x_petsc, comm)

    if rank == 0:
        assert x_global is not None

        true_rel_scipy = float(np.linalg.norm(A @ x_global - b) / np.linalg.norm(b))
        print("KSP converged reason: " f"{ksp.getConvergedReason()}")
        print("KSP iterations: " f"{ksp.getIterationNumber()}")
        print("PETSc distributed true relative residual: " f"{true_rel_petsc:.3e}")
        print("SciPy root cross-check residual: " f"{true_rel_scipy:.3e}")

        if _BENCH_MODE == "validate":
            print("iteration residuals:")
            print("  iter    PETSc rnorm       " "true ||Ax-b||/||b||")

            if len(residual_history) <= 30:
                rows = residual_history
            else:
                rows = (residual_history[:12] + [None] + residual_history[-12:])

            for row in rows:
                if row is None:
                    print("   ...")
                    continue
                (iteration, petsc_rnorm, true_rel_i) = row
                print(f"  {iteration:4d}    " f"{petsc_rnorm:12.5e}    " f"{true_rel_i:12.5e}")

    # --------------------------------------------------------------
    # 10. Explicit PETSc cleanup.
    # --------------------------------------------------------------
    true_residual_work.destroy()

    # IS objects were explicitly created by this script.
    is_u.destroy()
    is_lam.destroy()

    ksp.destroy()

    x_petsc.destroy()
    b_petsc.destroy()
    A_petsc.destroy()


if __name__ == "__main__":
    main()
