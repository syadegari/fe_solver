"""Bounded, frozen small-KKT diagnostics; never runs FE/material integration.

Run in py3.14, under mpiexec for distributed tests. Results include the original
unscaled PETSc and SciPy residuals. The production benchmark is left unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--matrix', type=int, choices=(0, 1), default=0)
parser.add_argument('--rhs', type=int, choices=(0, 1), default=0)
parser.add_argument('--block', choices=('full', 'force', 'constraint', 'random-force'), default='full')
parser.add_argument('--scale', action='store_true')
parser.add_argument('--guess', choices=('zero', 'compatible'), default='zero')
parser.add_argument('--output', type=Path, required=True)
args, petsc_options = parser.parse_known_args()
# Reuse the supplied benchmark's matrix distribution and exact baseline PC.
# Its module initializes PETSc and parses the database/group command line.
sys.argv = [sys.argv[0], f'kkt_debug_t0_iter{args.matrix}.h5', '--group',
            f't_n=0.0/iteration={args.matrix}', *petsc_options]
import benchmark_petsc_kkt as bench
import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from petsc4py import PETSc
from time import perf_counter


def main():
    comm = MPI.COMM_WORLD
    K, C, A, _ = bench.load_problem(Path(f'kkt_debug_t0_iter{args.matrix}.h5'),
                                   f't_n=0.0/iteration={args.matrix}')
    _, C_rhs, _, b = bench.load_problem(Path(f'kkt_debug_t0_iter{args.rhs}.h5'),
                                       f't_n=0.0/iteration={args.rhs}')
    if C.shape != C_rhs.shape or sparse.linalg.norm(C - C_rhs) != 0:
        raise ValueError('RHS swap requires identical constraints and DOF ordering')
    n, m = K.shape[0], C.shape[0]
    if args.block == 'force':
        b[n:] = 0
    elif args.block == 'constraint':
        b[:n] = 0
    elif args.block == 'random-force':
        b[:n] = np.random.default_rng(20261011).standard_normal(n)
        b[n:] = 0
    norm = np.linalg.norm(b)
    if norm == 0:
        raise ValueError('Cannot normalize a zero RHS')
    # All tests have unit RHS norm, preserving each selected direction.
    b /= norm
    physical_A = bench.create_distributed_petsc_matrix(A, comm)
    physical_b = bench.create_distributed_vector(b, comm)
    physical_x = physical_b.duplicate()
    physical_residual = physical_b.duplicate()
    # Congruence scaling: x = D y; solve D A D y = D b.
    # This leaves all displacement and multiplier unknowns in the full KKT.
    kref = float(np.median(np.abs(K.diagonal())))
    d = np.ones(n + m)
    if args.scale:
        d[:n] = 1 / np.sqrt(kref)
        d[n:] = np.sqrt(kref)
    scaled_A = (sparse.diags(d) @ A @ sparse.diags(d)).tocsr()
    operator = bench.create_distributed_petsc_matrix(scaled_A, comm)
    rhs = bench.create_distributed_vector(d * b, comm)
    diagonal = bench.create_distributed_vector(d, comm)
    x = rhs.duplicate()
    x.set(0)
    if args.guess == 'compatible':
        # Minimum Euclidean-norm compatible correction, not total displacement.
        u0 = C.T @ spsolve((C @ C.T).tocsc(), b[n:])
        start, end = x.getOwnershipRange()
        x.getArray()[:] = (np.r_[u0, np.zeros(m)] / d)[start:end]
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOptionsPrefix('bench_')
    ksp.setOperators(operator)
    ksp.setType('lgmres')
    ksp.setGMRESRestart(33)
    is_u, is_lam = bench.configure_distributed_block_pc(ksp, n, m, comm)
    options = PETSc.Options()
    defaults = {
        'ksp_lgmres_augment': 3,
        'fieldsplit_u_ksp_type': 'preonly',
        'fieldsplit_u_pc_type': 'asm',
        'fieldsplit_u_pc_asm_type': 'basic',
        'fieldsplit_u_pc_asm_overlap': 0,
        'fieldsplit_u_sub_ksp_type': 'preonly',
        'fieldsplit_u_sub_pc_type': 'ilu',
        'fieldsplit_u_sub_pc_factor_levels': 0,
        'fieldsplit_u_sub_pc_factor_mat_ordering_type': 'natural',
        'fieldsplit_lambda_ksp_type': 'preonly',
        'fieldsplit_lambda_pc_type': 'jacobi',
    }
    for name, value in defaults.items():
        bench.set_petsc_option_default(options, 'bench_' + name, value)
    ksp.setPCSide(PETSc.PC.Side.RIGHT)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(rtol=1e-9, atol=0, max_it=5000)
    ksp.setInitialGuessNonzero(args.guess != 'zero')
    ksp.setFromOptions()
    rtol, atol, _, max_it = ksp.getTolerances()
    if ksp.getPCSide() != PETSc.PC.Side.RIGHT or ksp.getNormType() != PETSc.KSP.NormType.UNPRECONDITIONED:
        raise ValueError('Requires right preconditioning and unpreconditioned norm')
    threshold = max(rtol, atol)  # original RHS has unit norm
    history = []
    checks = []
    scaled_b_norm = rhs.norm()

    def true_residual(solution):
        physical_x.pointwiseMult(diagonal, solution)
        physical_A.mult(physical_x, physical_residual)
        physical_residual.axpy(-1, physical_b)
        return float(physical_residual.norm())

    def convergence(ksp_, iteration, rnorm):
        history.append([iteration, float(rnorm / scaled_b_norm)])
        if not np.isfinite(rnorm):
            return PETSc.KSP.ConvergedReason.DIVERGED_NANORINF
        # For scaling or nonzero guesses, never change the acceptance norm or
        # denominator. Explicitly check original equations before convergence.
        if rnorm <= rtol * scaled_b_norm or iteration >= max_it:
            solution = ksp_.buildSolution()
            actual = true_residual(solution)
            checks.append([iteration, actual])
            if actual <= threshold:
                return PETSc.KSP.ConvergedReason.CONVERGED_RTOL
        if iteration >= max_it:
            return PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT
        return PETSc.KSP.ConvergedReason.ITERATING

    ksp.setConvergenceTest(convergence)
    start = perf_counter()
    ksp.setUp()
    ksp.setUpOnBlocks()
    setup_seconds = comm.allreduce(perf_counter() - start, op=MPI.MAX)
    start = perf_counter()
    ksp.solve(rhs, x)
    solve_seconds = comm.allreduce(perf_counter() - start, op=MPI.MAX)
    actual = true_residual(x)
    full_x = bench.gather_vector_to_root(physical_x, comm)
    accepted = int(ksp.getConvergedReason()) > 0 and actual <= threshold
    if comm.rank == 0:
        residual = A @ full_x - b
        scipy_residual = float(np.linalg.norm(residual))
        accepted = accepted and scipy_residual <= threshold
        result = dict(matrix=args.matrix, rhs=args.rhs, block=args.block,
                      scale=args.scale, guess=args.guess, ranks=comm.size,
                      kref=kref, rhs_original_norm=float(norm),
                      rhs_block_norms=[float(np.linalg.norm(b[:n])), float(np.linalg.norm(b[n:]))],
                      iterations=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()),
                      petsc_residual=actual, scipy_residual=scipy_residual,
                      residual_block_norms=[float(np.linalg.norm(residual[:n])), float(np.linalg.norm(residual[n:]))],
                      setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                      accepted=accepted, options=options.getAll(), history=history,
                      original_residual_checks=checks)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'options', 'original_residual_checks')}))
    accepted = comm.bcast(accepted, root=0)
    ksp.destroy()
    for obj in (is_u, is_lam, physical_A, physical_b, physical_x, physical_residual,
                operator, rhs, diagonal, x):
        obj.destroy()
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
