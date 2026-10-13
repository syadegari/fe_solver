# PETSc KKT frozen-system diagnostic results

## Review and provenance

The diagnostic scripts and saved results were copied from the execution checkout
and reviewed on 2026-10-13. The existing `benchmark_petsc_kkt.py`, SuperLU bridge
sources, and installation notes matched the development checkout byte-for-byte;
no production FE changes were imported from this experiment. The independent
SciPy/SVD snapshot inspection was rerun locally in `py3.14` and reproduced the
reported matrix changes, numerical nullities, and direct residuals below.
The MPI experiments themselves were run on the execution machine, not rerun
locally. Raw logs remain local-only under the ignored `kkt_diagnostics/` folder.
The scripts expect the two iteration-0/1 captures in the current directory.
Both snapshots are retained with Git LFS; run `git lfs pull` before reproducing
the study from a fresh checkout.

## Findings

The frozen snapshots reproduce RHS-sensitive restart stagnation. On eight MPI
ranks, increasing LGMRES's total approximation space from 33 to 100 resolves
the iteration-zero failure without changing the KKT equations, preconditioner,
initial correction, or original `1e-9` residual tolerance. This is verified in
the supplied, unchanged `benchmark_petsc_kkt.py`, using the original RHS magnitude:
350 iterations, PETSc and SciPy true relative residuals approximately `9.699e-10`.

These are small frozen-system results, not an integrated FE run or evidence of
robustness on the 24³ case. No constitutive or production solver code changed.

All experiments used Conda `py3.14`, PETSc 3.25.5, real float64, eight MPI ranks,
one thread per rank, right preconditioning, and overlap **0**. The supplied
benchmark defaults to overlap 1, so the reproduction commands override it.
`kkt_diagnostics/baseline.log` contains the original benchmark's full KSPView:
LGMRES(33,3), classical Gram–Schmidt without refinement, Schur LOWER/SELFP,
eight ASM BASIC subdomains, local PREONLY/ILU(0)/natural, multiplier
PREONLY/Jacobi. Full logs, residual histories, options and independent final
residuals are retained in `kkt_diagnostics/`.

The first controlled experiment held each matrix fixed and swapped the captured
RHS. The diagnostic harness normalizes each selected RHS to unit norm. All table
residuals below are recomputed for the **original, unscaled** full KKT equations,
relative to that RHS norm. The original benchmark reproduction also confirms the
result is not an artifact of this normalization.

| Matrix | RHS | Iterations | True relative residual | Accepted |
| --- | --- | ---: | ---: | --- |
| Iteration 0 | Iteration 0 | 5000 | 0.379206 | No |
| Iteration 0 | Iteration 1 | 555 | 8.95432e-10 | Yes |
| Iteration 1 | Iteration 0 | 5000 | 0.379215 | No |
| Iteration 1 | Iteration 1 | 555 | 9.12578e-10 | Yes |
| Iteration 0 | Constraint block of RHS 0, force block zero | 5000 | 0.379206 | No |
| Iteration 0 | Force block of RHS 1, constraint block zero | 555 | 8.93379e-10 | Yes |
| Iteration 0 | Seeded random force, constraint block zero | 622 | 9.50913e-10 | Yes |

The failure follows the RHS direction in this experiment. It does not require a
change in tangent or singularity of the full KKT matrix. The force-only control
uses RHS 1 rather than amplifying the near-zero force noise in RHS 0. The random
control uses NumPy seed 20261011.

Independent snapshot measurements, recorded in `kkt_diagnostics/snapshots.json`:

| Measurement | Iteration 0 | Iteration 1 |
| --- | ---: | ---: |
| Force RHS norm | 2.49982e-11 | 4.24260e-5 |
| Constraint RHS norm | 8.48528e-3 | 1.80520e-13 |
| `||K-K.T||F / ||K||F` | 2.80602e-7 | 1.96579e-6 |
| Diagonal minimum / median / maximum | 430.216 / 2530.420 / 5448.407 | 430.213 / 2530.420 / 5448.410 |
| Numerical nullity of K at relative SVD threshold 1e-12 | 6 | 3 |
| Saved direct correction true relative residual | 5.65361e-11 | 7.44381e-15 |
| Independent SciPy sparse direct residual | 5.06493e-11 | 8.17352e-15 |

`||K1-K0||F / ||K0||F = 9.72554e-6`; `C1-C0` is exactly zero.
The eigenvalues of `C C.T` range from 0.0827243 to 5.19469, so C has full row rank.
K0 has six singular values below 1.75e-12, followed by 61.1053; its largest
singular value is 14500.9. K1 has three near-zero singular values, then
2.55015e-5, 5.86784e-4, and 6.12544e-4. Translation-vector residuals are around
3e-18 relative to `||K||F ||v||`. The three additional K0 modes are consistent
with rigid rotations, but no coordinate data was used to identify them as such.
For each detected nullspace basis N, C N has full column rank. These measurements
support constraint removal of the rigid-mode degeneracy; no symmetry or exact
inverse of K was assumed. Dense SVD is restricted to these deliberately small
2187-DOF displacement blocks. All KKT solves and distributed operators are sparse.

The next tests changed one feature at a time on A0/b0 unless marked otherwise:

| Change | Iterations | Original true relative residual | Accepted |
| --- | ---: | ---: | --- |
| Baseline LOWER, restart 33 | 5000 | 0.379206 | No |
| Always refine classical Gram–Schmidt | 5000 | 0.379206 | No |
| Restart 100 | 350 | 9.69904e-10 | Yes |
| Congruence block scaling, restart 33 | 850 | 8.01917e-10 | Yes |
| Compatible initial correction, restart 33 | 797 | 9.98843e-10 | Yes |
| FULL factorization, restart 33 | 5000 | 0.420209 | No |
| FULL factorization, restart 100 | 287 | 7.26721e-10 | Yes |
| Sparse LU of SELFP matrix, LOWER, restart 33 | 5000 | 1.33426e-3 | No |

The LU control uses PETSc redundant PREONLY/LU on the small nonsingular
`-C diag(K)^(-1) C.T` approximation, confirmed by KSPView in `selfp_lu.log`.
It does **not** invert K, form the true Schur complement, or provide a direct
fallback for the outer KKT solve. It improves the residual substantially but
does not pass. Cheap multiplier application time in previous benchmarks does
not establish that the multiplier approximation is numerically adequate.

The supported explanation is an interaction of RHS direction, short restart,
and the block preconditioner/Euclidean residual geometry. For a pure constraint
RHS, the first LOWER-preconditioned direction cannot reduce the residual because
its image lies in the force block. That explains the initial step, not permanent
stagnation. FULL's failure confirms missing upper coupling is insufficient as a
complete explanation. Increasing the retained space succeeds for both
factorizations. Refinement leaves the plateau unchanged, so orthogonality loss
is not supported as the dominant mechanism here. This agrees with PETSc's warning
that LGMRES augmentation generally does not cure a stalled GMRES space of the
same size; restart counts Krylov and augmentation directions together.
[PETSc LGMRES documentation](https://petsc.org/release/manualpages/KSP/KSPLGMRES/).

The scaling test sets `kref = median(abs(diag(K)))` and
`D = diag(kref^(-1/2) I_u, kref^(1/2) I_lambda)`, then solves
`(D A D) y = D b`, recovering `x = D y`. It retains every KKT unknown and coupling.
The force block becomes K/kref while both C couplings remain unchanged. With
these fixed homogeneous block PCs, this also changes the residual norm geometry
that restarted GMRES minimizes. It is a controlled numerical scaling experiment,
not a generally established choice of physical nondimensionalization.

Crucially, accepting the scaled norm alone would produce a false success against
the original tolerance: at iteration 520 it crosses its `1e-9` relative target,
but the original residual is still **1.79754e-6**. The harness rebuilds the
correction and checks the original equations, continuing to iteration 850.
Scaling must carry a correct variable map and independent original-equation
acceptance into any production implementation. The scaled history in its JSON
is labeled by the scaled equation norm; `original_residual_checks` records the
original equation checks separately.

The compatible guess is a correction, not the committed total displacement:
`delta_u0 = C.T (C C.T)^(-1) b_c`, `delta_lambda0 = 0`.
It makes the initial constraint residual zero, transferring the work to the
force residual. Its initial original residual is actually about **1115 times**
the RHS norm; convergence remains measured against `||b||`, not that enlarged
initial residual. The final tolerance is unchanged. It uses a sparse solve of
the full-row-rank constraint Gram matrix and never eliminates displacement DOFs.
Production use must explicitly call `setInitialGuessNonzero(True)` so PETSc
retains the correction guess.
[PETSc initial-guess documentation](https://petsc.org/release/manualpages/KSP/KSPSetInitialGuessNonzero/).

Recommended next steps, in order:

1. Use restart 100 as the smallest demonstrated change, retaining augmentation 3,
   all existing full-KKT checks and the selected overlap. The command below
   reproduces the successful small case in the unchanged benchmark. Run a bounded
   frozen large-system validation before using it in a long FE solve. Restart 100
   is evidence for this case, not a universal optimum.
2. If large systems still stall, test consistent block scaling with original
   equation validation, then the compatible correction guess independently.
   Both succeeded here; neither licenses a looser residual test. Do not silently
   accept a scaled norm or change the denominator to the initial residual.
3. If iteration counts remain mesh/partition dependent, develop a coarse
   correction that represents the displacement rigid/near-null modes together
   with their constraint coupling. K is demonstrably singular: a stronger
   local ILU, an exact K solve, or an ordinary Schur formula using K inverse
   cannot be presumed valid. A constraint-aware augmented block or a coupled
   coarse KKT correction is a candidate requiring its own verification.

Attach exact nullspace information only to the appropriate singular displacement
operator, and near-nullspace information to compatible multigrid components;
the corresponding vectors are not null vectors of the full constrained KKT.
For nonsymmetric singular solves, left/right compatibility also matters.
[PETSc exact-nullspace documentation](https://petsc.org/release/manualpages/Mat/MatSetNullSpace/),
[PETSc near-nullspace documentation](https://petsc.org/release/manualpages/Mat/MatSetNearNullSpace/).
SELFP approximates the Schur matrix with the inverse diagonal of K; changing
factorization to FULL does not make this approximation exact.
[PETSc Schur preconditioner documentation](https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/).

Reproduction, from this folder:

```bash
HWLOC_COMPONENTS=-gl conda run -n py3.14 mpiexec -n 8 python \
  benchmark_petsc_kkt.py kkt_debug_t0_iter0.h5 \
  --group t_n=0.0/iteration=0 \
  -bench_fieldsplit_u_pc_asm_overlap 0 \
  -bench_ksp_gmres_restart 100

conda run -n py3.14 python inspect_kkt_snapshots.py
conda run --no-capture-output -n py3.14 python run_kkt_diagnostics.py
conda run --no-capture-output -n py3.14 python run_kkt_diagnostics.py --stage followup
```

Each runner solve has a 30-second wall-time limit in addition to the 5000-iteration
limit. Expected nonconvergence is recorded as evidence with a failing solver
exit status. Timing includes extra diagnostic work in some variants and is not
a production performance comparison. This work does not establish mesh/rank
independence, optimal restart size, or success on the large integrated problem.
