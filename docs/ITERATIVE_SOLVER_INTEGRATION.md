# Iterative KKT solver integration

Branch: `feature/iterative-solvers`, starting from `d357800` on `main`.

Status: initial integration implemented; frozen-system diagnostics reviewed;
successful real PETSc/MPI FE runtime verification is pending.
The supplied prototype and installation/experiment notes have been reviewed. Both direct
backends and existing example decks remain unchanged. This document tracks the branch; `IMPLEMENTATION_SPEC.md` remains
the executable contract and `FORMULATION.tex` the mathematical reference.

## Motivation and initial evidence

The user reports the following frozen-system benchmark:

| Quantity | Reported value |
|---|---:|
| Displacement unknowns / multipliers | 46,875 / 5,406 |
| Complete KKT size / stored entries | 52,281 / 3,522,771 |
| PETSc version | 3.25.5 |
| MPI ranks / threads per rank | 16 / 1 |
| Method / preconditioner | LGMRES / distributed PCFIELDSPLIT with Schur split |
| Linear iterations / solve time | 2,807 / 3.11 s |
| True relative residual, also checked with SciPy | 9.955e-10 |

The reported direct SuperLU comparison took approximately 50 s. This motivates
integration, but is not an end-to-end FE performance claim. The supplied
[`native/petsc_kkt_solver_session_summary.md`](../native/petsc_kkt_solver_session_summary.md)
records tuning, rank sensitivity, and installation provenance. The parameterized
[`native/benchmark_petsc_kkt.py`](../native/benchmark_petsc_kkt.py) accepts the actual
HDF5 database and group rather than a fixed machine-specific filename.

## Proposed nonlinear convergence profile

Use this profile for an explicitly selected comparison run, not a silent change
to all existing examples or a backend-dependent default:

```toml
force_rtol = 1.0e-5
force_atol = 1.0e-6
check_displacement_increment = true
displacement_rtol = 3.0e-6
displacement_atol = 1.0e-10
constraint_rtol = 1.0e-10
constraint_atol = 1.0e-10
```

The old relative coefficients are 1e-8 and 1e-9. Abaqus/Standard's normal
nonlinear force and correction coefficients are 5e-3 and 1e-2. Their logarithmic
midpoints with ours are approximately 7.1e-6 and 3.2e-6; the proposed values round
these to convenient trial settings. See the [Abaqus controls reference](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-controls.htm).

This is a compromise between coefficients, not an accuracy equivalence: our
force scale uses maximum assembled forces instead of Abaqus's averaged flux,
and our correction scale uses total displacement instead of incremental
displacement. Keep the dimensional absolute floors, affine-constraint tolerances,
Newton method, line search, iteration limits, and time controls unchanged initially.
Retain tight reference results and existing verification criteria; compare stress,
transformation fraction, and beta histories before adopting this profile generally.

## Integration boundaries

- Keep the complete, potentially nonsymmetric KKT system and current material,
  assembly, committed/trial-state, and line-search contracts. A Schur
  preconditioner is not elimination of the FE displacement constraints.
- Extend the existing `fe_solver/linear_solver.py` setup/solve/close boundary with
  minimal changes. For PETSc, setup owns a matrix/KSP/PC context rather than an LU
  factor. Preserve per-attempt ownership, full-Newton refresh, modified-Newton
  reuse, and cleanup on success or failure.
- Load PETSc/MPI dependencies only when the iterative backend is selected;
  ordinary SciPy/SuperLU use must remain independent of them. Resources remain
  explicitly user-selected; do not infer all available ranks or threads.
- Review the prototype's MPI execution model before choosing a launch contract.
  Preserve the existing element-process option without accidentally running a
  separate complete FE analysis or element pool on every MPI rank. Do not add
  distributed FE assembly as part of this work.

## Linear accuracy and verification

Linear and nonlinear tolerances are separate controls. Initially retain the
prototype's approximately 1e-9 true relative linear residual target, subject to
review of its actual stopping settings. Do not infer a universal linear
tolerance from the nonlinear coefficient or add adaptive forcing yet.

For `A dx = b`, check the true residual `e = A dx - b`, not only PETSc's reported
residual norm, which can be preconditioned or estimated. See
[PETSc convergence documentation](https://petsc.org/release/manualpages/KSP/KSPConvergedDefault/).
The aggregate relative two-norm mixes force and displacement-constraint equations;
it does not guarantee either block's required accuracy or a comparable error in
the correction vector. Inspect force and constraint residual blocks against their
respective scales. Nonlinear equilibrium and constraint checks remain authoritative.

## Work sequence

1. Review supplied prototype, options, READMEs, and environment requirements;
   settle MPI ownership, launch, and setup/reuse behavior before implementation.
2. Add a selectable, lazily loaded backend and concise linear-solve diagnostics:
   iterations, convergence reason, true residual, and setup/solve timings.
3. Verify a small nonsymmetric KKT system against the direct reference, including
   block residuals and context reuse/cleanup; run bounded nonlinear comparisons.
4. Have the user launch larger CP comparisons. Keep tolerances identical between
   backends for a correctness comparison; study the relaxed nonlinear profile
   separately against the existing tight reference. Update specification and
   formulation only as actual implementation decisions are settled.

## Implemented execution contract

- `--solver-backend petsc_lgmres` is selectable alongside both direct backends.
- `mpiexec -n N` determines the persistent MPI team; only rank zero performs FE work.
  `--num-processes` determines rank zero's independent local element-worker pool.
  Use OpenMPI `--bind-to none` on the shared-memory machine, or allocate rank zero
  enough affinity for its workers. No worker initializes MPI.
- Root scatters CSR row blocks at setup, rather than replicating the complete SciPy
  matrix on every rank as the frozen-system benchmark does. RHS distribution and
  correction gathering happen once per solve; all Krylov/PC work stays distributed.
- The separate packaged `fe_solver/petsc_lgmres.toml` provides native PETSc options;
  `--petsc-options FILE` overrides them. Resolved options/version/ranks are logged.
  Overlap 1 remains the recommended baseline. Test overlap 0 against 1 on the same
  frozen matrix/rank count before using the tuned option for a nonlinear study.
- Existing setup/solve/close lifetimes are retained. PETSc uses the existing setup
  timing field, historically named `kkt_factorization_wall_seconds`, without
  pretending its work is a direct factorization.
- Explicit final linear-residual validation and positive convergence reason are
  mandatory. Ordinary nonconvergence requests rollback/cutback; no implicit direct
  fallback is implemented. Unexpected collective exceptions abort the MPI job.

## Verification checkpoint

Local `py3.14` currently has no PETSc, mpi4py, or MPI launcher. Do not interpret mocked
routing/lifecycle tests as numerical PETSc validation. The bounded real checks are:

```bash
python -m unittest tests.test_petsc_solver tests.test_linear_solver
mpiexec --bind-to none -n 2 python -m verification.check_petsc_solver
```

The MPI check compares nonsymmetric KKT corrections, repeated/zero RHS, overlap 0/1,
and direct-versus-iterative HDF5/reaction results for both Newton variants with two
rank-zero element workers. It writes artifacts to a temporary directory by default.
Run it at one and two ranks in the installed environment. It is a bounded integration
gate, not evidence of overlap robustness on the large periodic matrix. After that,
compare the supplied approximately 3M-entry matrix at overlaps 0 and 1 with identical
rank count/tolerances, then run a bounded stateful periodic/J2 case before long CP runs.
The relaxed nonlinear profile above remains a separate comparison axis.

The user reported cutback exhaustion on the 24³ TRIP simple-shear case at 4, 8,
and 16 ranks. The old controller discarded the underlying exception and did not
save the final failed attempts. Reporting now preserves that cause and flushes
failed-attempt Newton/PETSc diagnostics before retry or termination; numerical
settings are unchanged. The rerun identified KSP nonconvergence at iteration
zero: 5000 iterations and true relative residual approximately 0.179, unchanged
as cutbacks halved the RHS. No Newton correction or line-search trial was accepted.
The subsequent small frozen-system investigation below isolates RHS-sensitive
restart stagnation; its remedy still needs validation on the large system and
in integrated FE time stepping.

### Temporary first-increment capture

`fe_solver/solver_capture.py` is a drop-in copy of `solver.py` with diagnostic
HDF5 capture only. Temporarily replace the remote `solver.py` with this sibling
file, retain the original, and run the direct backend to advance past iteration
zero. For example, in the activated project environment:

```bash
python -m fe_solver solve PATH_TO_DECK --solver-backend superlu_mt \
    --num-threads 16 --num-processes 16 --debug-timing --stop-time 0.1
```

Restore the original solver afterwards. The first attempted increment at `t_n=0`
produces these files under the deck's resolved output directory:

- `kkt_capture/kkt_debug_t0_iter0.h5`, group `t_n=0.0/iteration=0`;
- `kkt_capture/kkt_debug_t0_iter1.h5`, group `t_n=0.0/iteration=1`.

Each group contains CSR `K` and `C` (`data`, `indices`, `indptr`, `shape` attribute)
and the unnegated `r_u`, `r_c` vectors, exactly as in the existing benchmark
databases. Successful solves also add the unscaled `correction` vector. The
extra `t_np1` attribute identifies the attempted endpoint. Both benchmark readers
accept this layout unchanged. Modified Newton captures its actual frozen tangent.
Retries do not replace the first-attempt captures, and an existing capture file
is never overwritten: choose a fresh output directory for a new capture run.

The first failing solve was nearly constraint-only. Replay a captured matrix with
different RHS block patterns and compare lower/full Schur factorizations before
attributing failure to the material model or relaxing convergence criteria.

The user replayed the 8³ captures with overlap 0. At 8/16 ranks, iteration zero
failed after 5000 iterations with lower factorization (true relative residuals
0.379/0.309), while iteration one converged in 555/518 iterations. Full
factorization at 16 ranks still failed on iteration zero (0.298), but reduced
iteration one's count to 320. A standalone, data/code-independent research brief
records the configuration, results, and unresolved hypotheses in
[`PETSC_KKT_STAGNATION_BRIEF.md`](PETSC_KKT_STAGNATION_BRIEF.md).

### Reviewed frozen-system investigation

The remote `native` sources were compared with the development checkout on
2026-10-13. The benchmark and SuperLU bridge sources were unchanged. Imported
additions are `diagnose_petsc_kkt.py`, `inspect_kkt_snapshots.py`,
`run_kkt_diagnostics.py`, and
[`PETSC_KKT_DIAGNOSTIC_RESULTS.md`](../native/PETSC_KKT_DIAGNOSTIC_RESULTS.md).
Generated diagnostic logs remain ignored. The research brief above is retained
as the pre-investigation problem statement, not the latest findings.

On the small 8³ snapshots, swapping RHS vectors while holding each matrix fixed
showed that failure follows the nearly constraint-only RHS, not the small tangent
change. Classical Gram-Schmidt refinement did not resolve stagnation. Restart
100 passed with both LOWER and FULL Schur factorizations; consistent block scaling
and a compatible correction guess also passed with restart 33, using the original
equation residual for acceptance. The independent local SciPy/SVD inspection
reproduced the report's direct residuals and six/three numerical null modes of K.
These observations do not justify assuming K is invertible or eliminating constraints.

The packaged solver still uses restart 33 and zero correction guesses: this is a
checkpoint of the current implementation, not an implicit adoption of experimental
settings. Use a separate override containing `ksp_gmres_restart = 100` for the next
bounded validation, keeping the original tolerance and selected overlap unchanged.
Remaining gates are the one-/two-rank bounded MPI integration checks, a frozen
large-system test, and a bounded stateful FE comparison before a long TRIP run.

At the 2026-10-13 checkpoint, all nine local unittest modules ran in `py3.14`:
92 tests total, 90 passed and two dependency-related tests skipped. The new
diagnostic scripts also passed Python compilation. PETSc/MPI numerical execution
is unavailable locally; mock routing/lifecycle tests are not a substitute for
the remaining real integration gates.
