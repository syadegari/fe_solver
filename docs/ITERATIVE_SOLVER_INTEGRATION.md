# Iterative KKT solver integration

Branch: `feature/iterative-solvers`, starting from `d357800` on `main`.

Status: integration preparation. The user's PETSc prototype and READMEs are pending;
no iterative backend is registered yet. Existing solver behavior and example decks
are unchanged. This document tracks the branch; `IMPLEMENTATION_SPEC.md` remains
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
integration, but is not an end-to-end FE performance claim. Exact options,
preconditioner details, and reproduction commands will come from the supplied
prototype rather than being reconstructed from the console summary.

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
