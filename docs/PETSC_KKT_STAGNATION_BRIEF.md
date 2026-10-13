# PETSc LGMRES stagnation on constraint-driven FE KKT systems

## Request to another assistant

Analyze the problem below and propose a short, prioritized diagnostic sequence
and plausible remedies. You have no access to the matrices, application code,
or constitutive routines. Separate observations from hypotheses; use primary
references, preferably PETSc documentation/source and numerical linear-algebra
literature. The complete KKT formulation and independent true-residual checks
must remain intact. Do not suggest relaxing tolerances merely to accept an
unconverged solution. Correctness is the priority, not performance tuning.

## 1. Mathematical problem

A quasistatic finite-strain FE solver imposes affine displacement constraints
using Lagrange multipliers. Its Newton correction solves

$$
A\,\Delta x=b,\qquad
A=\begin{bmatrix}K&C^T\\C&0\end{bmatrix},\qquad
\Delta x=\begin{bmatrix}\Delta u\\\Delta\lambda\end{bmatrix},\qquad
b=-\begin{bmatrix}r_u\\r_c\end{bmatrix},
$$

with

$$
r_u=f_{\mathrm{int}}(u)-f_{\mathrm{ext}}+C^T\lambda,
\qquad r_c=Cu-d(t).
$$

`K` is the assembled material/geometric displacement tangent. Symmetry is not
assumed. `C` is a constant sparse constraint matrix imposing periodic displacement
jumps and a three-component anchor. Its rows are predominantly differences of
two DOFs with coefficients `+1,-1`, plus three single-entry anchor rows.
No displacement DOFs are eliminated.

The unconstrained displacement block can contain rigid-body modes even when
the complete KKT system is nonsingular. **Do not assume that `K` is invertible.**
Its numerical nullspace, symmetry error, spectrum, and diagonal ranges have not
yet been measured.

Each increment starts global Newton from the committed `u_n, lambda_n`, without
an affine displacement predictor. At cold start these are zero and stress is
approximately zero. A change in prescribed periodic deformation therefore gives
`r_u ≈ 0`, `r_c != 0`: the first linear solve is nearly constraint-only.

Later Newton iterates have a force residual. Their RHS need not be fully
populated: a full correction can satisfy the linear constraints exactly, leaving
the next constraint residual near zero. Separate RHS block norms for the
successful iteration-one capture have not been reported.

The FE problem uses orientation-dependent ferritic/austenitic multiphase TRIP
steel and fully integrated Hex8 elements with centroidal F-bar projection.
The large case prescribes periodic simple shear to `F12=0.2` over physical time
`[0,2000]`, starting with `dt=0.1`. The frozen-system benchmarks below perform
**no constitutive update, Newton iteration, line search, or time stepping**.

## 2. Solver and execution settings

- Ubuntu 24.04; PETSc/petsc4py **3.25.5**; real `float64` scalars, `int32` indices.
- Distributed MPIAIJ matrix with approximately equal contiguous row partitions.
- Global ordering is all displacement DOFs followed by all multipliers.
  Distributed index sets explicitly define the two fields.
- One BLAS/OpenMP thread per MPI rank.
- Zero Krylov initial guess for the **correction**, not total displacement.
- Fixed linear block PCs applied with `preonly`, not tolerance-controlled inner
  Krylov iterations.
- No explicit equation/block scaling or nondimensionalization is applied.
  The aggregate Euclidean norm mixes force equations and displacement constraints.

Native PETSc options are listed below. The standalone script prefixes them with
`bench_`. The small experiments use overlap **0**; the original large integrated
failure used overlap **1**.

```text
ksp_type                                      lgmres
ksp_pc_side                                   right
ksp_norm_type                                 unpreconditioned
ksp_rtol                                      1e-9
ksp_atol                                      0
ksp_max_it                                    5000
ksp_gmres_restart                             33
ksp_lgmres_augment                            3
pc_type                                       fieldsplit
pc_fieldsplit_type                            schur
pc_fieldsplit_schur_fact_type                  lower (also tested full)
pc_fieldsplit_schur_precondition               selfp
fieldsplit_u_ksp_type                         preonly
fieldsplit_u_pc_type                          asm
fieldsplit_u_pc_asm_type                      basic
fieldsplit_u_pc_asm_overlap                   0
fieldsplit_u_sub_ksp_type                     preonly
fieldsplit_u_sub_pc_type                      ilu
fieldsplit_u_sub_pc_factor_levels             0
fieldsplit_u_sub_pc_factor_mat_ordering_type  natural
fieldsplit_lambda_ksp_type                    preonly
fieldsplit_lambda_pc_type                     jacobi
```

`selfp` assembles an approximate Schur preconditioning matrix using the diagonal
inverse of the displacement block, here approximately

$$
S_p=-C\operatorname{diag}(K)^{-1}C^T.
$$

The multiplier PC is Jacobi on this approximation. Changing Schur
**factorization** from `lower` to `full` does not replace `selfp` with an exact
Schur complement. See [PETSc PCFIELDSPLIT](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/)
and [Schur factorization types](https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurFactType/).

The standalone benchmark's convergence callback checks the monitored norm
against `max(rtol*||b||2, atol)` and enforces 5000 iterations. After solving,
`||A*x-b||2/||b||2` is independently recomputed with distributed PETSc operations
and with the gathered correction and a root-side SciPy matrix. Both agree with
the monitored norm in the reported experiments.

## 3. Small frozen systems: 8³ elements

A successful direct SuperLU_MT FE run captured global Newton iterations **0 and
1 of the first increment**, both with `t_n=0`. Both `K` and the RHS can change
between these captures. They are **not yet a fixed-matrix/RHS-only experiment**.

Both systems have:

| Matrix | Shape | Stored entries |
|---|---:|---:|
| `K` | 2187 × 2187 | 140625 |
| `C` | 654 × 2187 | 1305 |
| `A` | 2841 × 2841 | 143235 |

Initial RHS norms: iteration 0, `8.48528e-3`; iteration 1, `4.24260e-5`.

All results below use overlap 0 and identical tolerances, restart/augmentation,
and local ILU/Jacobi settings:

| Capture | MPI ranks | Schur factorization | Iterations | Reason | True relative residual |
|---|---:|---|---:|---:|---:|
| Iteration 0 | 8 | lower | 5000 | -3 | 3.79206e-1 |
| Iteration 0 | 16 | lower | 5000 | -3 | 3.09073e-1 |
| Iteration 0 | 16 | full | 5000 | -3 | 2.98147e-1 |
| Iteration 1 | 8 | lower | 555 | 2 | 9.12578e-10 |
| Iteration 1 | 16 | lower | 518 | 2 | 9.85911e-10 |
| Iteration 1 | 16 | full | 320 | 2 | 9.66568e-10 |

Reason `-3` is iteration-limit exhaustion; `2` is relative-residual convergence.
See [PETSc convergence reasons](https://petsc.org/release/manualpages/KSP/KSPConvergedReason/).

For iteration zero, early Krylov iterations barely reduce the residual and the
printed final iterations are flat. At 16 ranks:

| Krylov iteration | Relative residual, lower | Relative residual, full |
|---:|---:|---:|
| 0 | 1.00000 | 1.00000 |
| 1 | 1.00000 | 0.999999 |
| 2 | 0.999999 | 0.999999 |
| 11 | 0.999950 | 0.999699 |
| 5000 | 0.309073 | 0.298147 |

Iteration one improves immediately: its first Krylov iteration gives relative
residual `0.489188` with lower and `0.351341` with full. Independently recomputed
PETSc and SciPy residuals agree, including for failures. There is no observed
residual-estimate/true-residual mismatch.

## 4. Original large integrated failure: 24³ elements

The FE application failed before accepting a Newton correction at 4, 8, and 16
MPI ranks. Detailed available diagnostics are for 16 ranks, overlap 1:

| Matrix | Shape | Stored entries |
|---|---:|---:|
| `K` | 46875 × 46875 | 3501153 |
| `C` | 5406 × 46875 | 10809 |
| `A` | 52281 × 52281 | 3522771 |

At Newton iteration zero:

- `||r_u||inf ≈ 9.72e-13`, `||r_c||inf = 1e-3`.
- `||b||2 ≈ 2.44949e-2`.
- After 5000 Krylov iterations: true relative residual `0.1792801141`.

Cutbacks from `dt=0.1` to `0.05`, `0.025`, `0.0125`, and `0.00625` halved absolute
residuals but left this relative residual essentially unchanged. This suggests
an effectively unchanged matrix/preconditioner with a scaled RHS; equality of
the matrices between retries has not been measured. No Newton correction was
accepted, and no line-search candidate was evaluated.

A previous benchmark on a different J2-matrix/hyperelastic-core system with the
same large dimensions and sparsity counts, captured at a later Newton iteration,
converged in 2807 iterations at 16 ranks with overlap 0 and true relative residual
approximately `1e-9`. Identical sizes do not establish identical matrices.

## 5. Current conclusions and unresolved points

Established:

- The iteration-zero failure reproduces outside FE integration on a frozen system.
- The direct FE backend advances through iteration zero and produces iteration
  one. Quantified direct correction errors/residuals are not supplied here.
- Both lower and full fail on iteration zero, and both converge on iteration one.
  Missing upper coupling alone therefore does not explain the persistent failure.
- The observed failure precedes Newton/line-search decisions; it is not rejection
  of a converged linear solution by the additional residual-validation check.

Unresolved:

- Whether the RHS pattern alone explains the difference: `K` has not been held
  fixed while swapping RHS vectors.
- Whether equation scaling, short restart, loss of orthogonality, displacement
  near-nullspaces, local ILU quality, or the Schur approximation is dominant.
- Actual nested KSP/PC settings have not been supplied as a `KSPView` dump;
  the listed settings are the requested configuration.
- Matrix ranks, spectra, symmetry errors, and diagonal/block norm ranges are not
  available. A constraint-only RHS cannot itself change matrix rank.

## 6. Analytical clue, not a complete explanation

Let `H` be the approximate displacement inverse and `G` the approximate Schur
inverse. With right preconditioning and lower factorization, an exactly
constraint-only RHS has preconditioner action

$$
z_u=H\,0=0,\qquad z_\lambda=G b_c,
\qquad Az=\begin{bmatrix}C^T G b_c\\0\end{bmatrix}.
$$

The first residual-change direction is supported only in the force block and
is orthogonal to the initial constraint-only residual in the unweighted Euclidean
inner product. The first GMRES minimization cannot reduce the residual in exact
arithmetic. Later Krylov directions introduce displacement components, so this
does **not** imply permanent stagnation or singularity. Full factorization adds
displacement coupling, but the observed full-factorization failure shows that
this first-direction explanation is insufficient.

PETSc's [LGMRES documentation](https://petsc.org/release/manualpages/KSP/KSPLGMRES/)
cautions that augmentation generally does not cure stalled restarted GMRES.
Its restart setting counts the total approximation space, including augmentation:
the requested size is 33 total, not 33 plus 3. Restart size and augmentation have
not yet been varied.

## 7. Questions to answer

1. What mechanisms could explain constraint-driven stagnation for both lower and
   full Schur factorizations, while a later Newton system converges?
2. What minimal fixed-matrix/RHS test separates RHS sensitivity from a change in
   tangent? Consider captured RHS swaps and nonzero force-only/constraint-only
   controls, normalized to comparable RHS norms.
3. Which measurements best distinguish restart stagnation, orthogonality loss,
   poor block scaling, and weak block preconditioning?
4. How should rigid-body modes of the unconstrained `K` be treated? Do not propose
   an exact inverse of this block without accounting for its possible singularity.
5. Which small controlled changes should be tried first, and what observation
   would support each explanation? Preserve independent true-residual validation.
6. Could a constraint-compatible correction initial guess help without changing
   the full-KKT formulation? Distinguish corrections from total `u_n`: global
   Newton already starts at `u_n`; Krylov starts at zero correction.

Please provide a reasoned sequence, not a large unranked sweep of PETSc knobs.
No constitutive changes, displacement elimination, automatic direct fallback,
or nonlinear tolerance relaxation is requested.
