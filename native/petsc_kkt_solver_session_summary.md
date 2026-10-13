# PETSc / MPI KKT Solver Optimization — Session Handoff Summary

## Purpose

This document summarizes the entire solver investigation from the original KKT benchmark through the latest PETSc/MPI tuning results. It is intended to be pasted or uploaded into a future ChatGPT session so work can continue without losing the technical history.

Repository integration note: the original `petsc_mpi_hybrid_bench_1.py` is now
`benchmark_petsc_kkt.py`. Supply the database and `--group` explicitly; historical
commands below require those additional arguments. The numerical history below is
retained as experiment evidence, not rerun integration results.

---

## 1. Problem and machine

The linear system is a saddle-point / KKT system

$$
A =
\begin{bmatrix}
K & C^T \\
C & 0
\end{bmatrix},
\qquad
b =
-\begin{bmatrix}
r_u \\
r_c
\end{bmatrix}.
$$

The main "large" benchmark used throughout the later tuning has:

- `K`: `(46875, 46875)`, `nnz = 3,501,153`
- `C`: `(5406, 46875)`, `nnz = 10,809`
- full KKT matrix `A`: `(52281, 52281)`, `nnz = 3,522,771`
- scalar type: `float64`
- PETSc integer type: `int32`
- PETSc version during the tests: `3.25.5`
- machine: 16 physical CPU cores
- MPI tests used `BENCH_NUM_THREADS=1`, so these were effectively pure-MPI tests rather than hybrid MPI/OpenMP tests.

A smaller benchmark used earlier had:

- `K`: `(14739, 14739)`, `nnz = 1,058,841`
- `C`: `(2454, 14739)`, `nnz = 4,905`
- `A`: `(17193, 17193)`, `nnz = 1,068,651`

The Python/HDF5 loading path currently keeps a replicated SciPy copy of the input matrices on every MPI rank for simplicity and validation, while the PETSc matrix/vector, Krylov solver, and preconditioner are distributed. This replicated input design will eventually become a memory-scaling limitation for much larger cases.

---

## 2. Original approach and why it was problematic

The early approach used SciPy/SuperLU-style components and a much heavier block preconditioner. A representative SciPy block-triangular preconditioner used:

- `spilu(K)` as an approximate inverse of the `K` block
- an explicitly assembled approximate Schur complement
 $$
  \hat S = C \hat K^{-1} C^T
  $$
- dense LU factorization of that Schur complement
- SciPy `lgmres`

An earlier root-centric MPI version also used global/root SciPy factorization plus gather/scatter operations around preconditioner applications. That design was fundamentally non-scalable because rank 0 remained responsible for global work and communication.

For the large problem, the older SuperLU_MT-based method took about **55 s using 16 threads**. The important later observation was that SuperLU_MT's solve phase was negligible (less than about 0.1% of its workload); the bottleneck was factorization, especially severe fill-in. On a smaller test, SuperLU_MT was around **3.3 s** for factor/solve.

This became the key contrast with PETSc: PETSc eventually won by deliberately using a much cheaper approximate factorization and tolerating more Krylov iterations.

---

## 3. Distributed PETSc formulation

The scalable PETSc version uses the full KKT matrix with:

- outer `KSP`: `LGMRES`
- `PCFIELDSPLIT`
- Schur-complement split
- Schur factorization type: `LOWER`
- Schur preconditioner type: `SELFP`
- `u/K` field: initially `PREONLY + ASM(overlap=1)`, local subsolver `PREONLY + ILU(1)`
- `lambda/Schur` field: `PREONLY + Jacobi`

Conceptually:

$$
P^{-1}
\approx
\begin{bmatrix}
\hat K^{-1} & 0 \\
* & \hat S^{-1}
\end{bmatrix},
$$

with the Schur preconditioner based on

$$
\hat S \approx -C\,\mathrm{diag}(K)^{-1} C^T.
$$

A major discovery was that this cheap Schur approximation is extremely inexpensive in practice and is not the bottleneck.

---

## 4. Correct residual semantics were essential

An early PETSc version used LEFT preconditioning and PETSc's PRECONDITIONED norm, while the custom callback independently checked the physical residual. That made PETSc's convergence criterion and the callback's convergence criterion inconsistent.

The solver was changed to:

```python
ksp.setPCSide(
    PETSc.PC.Side.RIGHT
)

ksp.setNormType(
    PETSc.KSP.NormType.UNPRECONDITIONED
)
```

This makes PETSc's reported residual norm the physical residual norm for the original system.

The validate callback was then simplified from rebuilding the solution and explicitly computing `A*x-b` every iteration to:

```python
true_abs = float(
    petsc_rnorm
)

true_rel = (
    true_abs
    / b_norm
)
```

Only one independent distributed `A*x-b` residual is computed after the solve, followed by one SciPy root residual after gathering the final solution.

This callback optimization was significant. On an earlier smaller 4-rank case, runtime dropped from roughly **2.84 s to 1.47 s** while preserving the same **906 iterations** at the then-used `1e-8` tolerance.

The later benchmark defaults became approximately:

- `rtol = 1e-9`
- `max_it = 5000`
- LGMRES restart `33`
- LGMRES augmentation dimension `3`
- RIGHT preconditioning
- UNPRECONDITIONED norm

---

## 5. Early MPI scaling showed that the numerical preconditioner, not MPI execution, was the main problem

With the initial one-level ASM + ILU(1) setup, individual iterations got cheaper as ranks increased, but iteration counts often grew badly or became pathological.

### Smaller problem, `rtol=1e-9`, default ASM behavior

- 1 rank: 784 iterations, 1.72 s
- 2 ranks: 1577 iterations, 4.21 s
- 3 ranks: hit 5000 iterations and stalled around `8.1e-7`
- 4 ranks: 1247 iterations, 1.93 s
- 8 ranks: 1636 iterations, 2.52 s
- 16 ranks: 1672 iterations, 2.67 s

The 3-rank decomposition was especially pathological.

### Large problem, older `rtol=1e-8`, `max_it=2000`

- 1 rank: 915 iterations, 9.69 s, converged
- 2 ranks: 2000 iterations, 18.94 s, failed tolerance
- 4 ranks: 2000 iterations, 14.77 s, failed tolerance
- 8 ranks: 2000 iterations, 13.94 s, failed tolerance
- 16 ranks: 1609 iterations, 11.50 s, converged

The approximate per-iteration costs nevertheless improved from about 10.6 ms at 1 rank to about 7.0 ms at 8 ranks. This was the first strong indication that the distributed kernels themselves were scaling, while one-level ASM quality was changing with the process decomposition.

---

## 6. `ASM BASIC` removed some catastrophic partition behavior

The PETSc ASM default restriction/interpolation behavior was suspected of contributing to partition-layout sensitivity.

A large-system sweep using:

```bash
-bench_fieldsplit_u_pc_asm_type basic
```

with ILU(1), `rtol=1e-9`, `max_it=5000` produced:

- 1 rank: 1151 it, 11.88 s
- 2 ranks: 2731 it, 25.57 s
- 3 ranks: 1774 it, 14.62 s
- 4 ranks: 2705 it, 19.91 s
- 5 ranks: 2200 it, 16.40 s
- 6 ranks: 2302 it, 16.66 s
- 7 ranks: 2435 it, 17.32 s
- 8 ranks: 2599 it, 18.17 s
- 16 ranks: 3000 it, 21.49 s

All these cases converged. BASIC therefore improved robustness, but did not by itself provide scalable time-to-solution.

The per-iteration cost still improved modestly through about 8 ranks, while total time was dominated by increased iteration counts.

---

## 7. Fixing the number of ASM blocks separated execution scaling from preconditioner changes

A diagnostic test held the requested ASM block count at 8 while varying MPI ranks:

```bash
-bench_fieldsplit_u_pc_asm_type basic
-bench_fieldsplit_u_pc_asm_blocks 8
```

Results:

- 1 rank: 2401 it, 39.95 s
- 2 ranks: 2104 it, 24.02 s
- 4 ranks: 2135 it, 17.00 s
- 8 ranks: 2599 it, 18.18 s

Approximate cost per iteration:

- 1 rank: 16.64 ms
- 2 ranks: 11.42 ms
- 4 ranks: 7.96 ms
- 8 ranks: 6.99 ms

This established that PETSc's distributed work was indeed parallelizing. The total runtime remained poor because convergence still depended strongly on how the one-level subdomains were realized.

This test was **not** a two-level domain decomposition method. It was still one-level ASM; there was no global coarse correction.

---

## 8. Two-level domain decomposition remained the long-term robust-scaling direction

A genuine two-level method would combine local subdomain corrections with a global coarse correction, schematically:

$$
M^{-1}_{2L}
=
M^{-1}_{local}
+
Z(Z^T K Z)^{-1}Z^T.
$$

That coarse term is what one-level ASM lacks.

PETSc candidates discussed for future work were:

- `PCHPDDM`: overlapping multilevel/spectral domain decomposition; conceptually closest to the present ASM approach, but requires more than simply switching `pc_type`
- `PCBDDC`: two-level non-overlapping domain decomposition, typically benefiting from richer finite-element/domain metadata
- multigrid alternatives such as PETSc GAMG or HYPRE BoomerAMG, depending on `K`'s mathematical structure

No two-level method has yet been implemented or benchmarked in this session.

---

## 9. Attempting to reproduce the PETSc algorithm in SciPy performed very poorly

A SciPy analogue was constructed with:

- explicit RIGHT preconditioning by solving
$$
  A P^{-1}y=b,\quad x=P^{-1}y
$$
- SuperLU `spilu(K)`
- the cheap PETSc-like `SELFP + Jacobi` Schur approximation rather than the old dense Schur LU
- SciPy `lgmres`

The result was:

- SuperLU ILU factorization: **43.04 s**
- factor stored nnz: **9,970,616**
- fill ratio: **2.848**
- right-preconditioned SciPy LGMRES: **76.88 s**
- final residual about `9.99e-10`
- total roughly **120 s**

This showed that the PETSc advantage was not merely the block formulation. PETSc's native ILU(k) implementation is dramatically better suited to this matrix than SciPy/SuperLU's threshold/drop-based `spilu`.

Important distinction:

- PETSc `PCILU` with `pc_factor_levels = k` is a level-of-fill ILU(k)
- SciPy `spilu()` is SuperLU ILU with drop tolerance, fill factor, drop rules, etc.
- they are not equivalent algorithms

Therefore a stock SciPy-only implementation cannot reproduce PETSc ILU(0)/ILU(1) exactly.

---

## 10. `-log_view` explained where PETSc spends its time

For the 1-rank ILU(1) baseline, `-log_view` showed approximately:

- total PETSc stage: 12.44 s
- `KSPSolve`: 11.50 s
- `MatSolve`: **8.11 s** (~65% of total)
- `MatMult`: **2.88 s** (~23%)
- LGMRES orthogonalization: ~0.31 s
- symbolic ILU factorization: ~0.39 s
- numeric factorization: ~0.25 s
- block setup total: ~0.68 s
- Schur field solve: only about **0.005 s**

The major finding was that PETSc does **not** spend meaningful time building the ILU. It builds ILU(1) in well under one second. Runtime is dominated by repeatedly applying the triangular solve.

This explains why reducing fill became much more valuable than reducing iteration count.

---

## 11. Benchmark timing itself was corrected

Originally the program reported roughly:

```text
PETSc distributed KSP/PC setup: ~0.03 s
PETSc distributed LGMRES solution: ~12.18 s
```

but the nested block factorization was actually being lazily set up inside the solve.

The code was changed so setup explicitly includes block setup, using `KSP.setUpOnBlocks()` after `ksp.setUp()`.

After this change, the same ILU(1) case reported approximately:

- setup: ~0.70 s
- solve: ~11.24 s

This is the correct accounting and should be retained.

---

## 12. ILU(1) ordering sweep: natural ordering was best overall

At 1 rank with ILU(1):

| Ordering | Factor nnz | Fill ratio | Setup | Iterations | Solve | Setup+solve |
|---|---:|---:|---:|---:|---:|---:|
| natural | 8,384,769 | 2.395 | 0.70 s | 1151 | 11.24 s | 11.94 s |
| rcm | 7,645,275 | 2.184 | 0.62 s | 1543 | 14.30 s | 14.92 s |
| qmd | 9,727,659 | 2.778 | 1.06 s | 1609 | 17.35 s | 18.41 s |
| 1wd | 7,645,275 | 2.184 | 0.60 s | 1543 | 14.35 s | 14.95 s |
| nd | 9,707,751 | 2.773 | 0.98 s | 1407 | 14.91 s | 15.89 s |

RCM/1WD made each iteration slightly cheaper by reducing factor size, but convergence degraded enough that total time increased.

Conclusion: **natural ordering is the best tested ordering for this matrix.**

---

## 13. ILU level sweep produced the single biggest tuning improvement

At 1 rank, natural ordering, restart 33:

| ILU level | Factor nnz | Fill ratio | Setup | Iterations | Solve | Setup+solve |
|---:|---:|---:|---:|---:|---:|---:|
| **0** | **3,501,153** | **1.000** | **0.14 s** | 1477 | **6.86 s** | **7.00 s** |
| 1 | 8,384,769 | 2.395 | 0.71 s | 1151 | 11.36 s | 12.07 s |
| 2 | 15,859,215 | 4.530 | 2.26 s | 1001 | 15.68 s | 17.94 s |

Approximate solve cost per iteration:

- ILU(0): 4.64 ms
- ILU(1): 9.87 ms
- ILU(2): 15.66 ms

This was a critical finding: **minimizing Krylov iterations is not the objective; minimizing total wall time is.**

ILU(0) requires more iterations but makes each preconditioner application so much cheaper that it wins decisively.

The program default was therefore changed from ILU(1) to:

```text
bench_fieldsplit_u_sub_pc_factor_levels = 0
```

---

## 14. LGMRES restart sweeps: restart 33 remained the best tested value

### With ILU(1)

- restart 33: 1151 it, 11.32 s
- restart 50: 1110 it, 11.55 s
- restart 75: 1187 it, 12.90 s
- restart 100: 1092 it, 12.38 s

Larger restart sometimes reduced iteration count but increased per-iteration orthogonalization/storage cost enough to lose overall.

### With ILU(0)

- restart 20: 2830 it, 12.29 s
- restart 25: 1912 it, 8.56 s
- **restart 33: 1477 it, 6.92 s**
- restart 40: 1712 it, 8.17 s
- restart 50: 1492 it, 7.36 s

Conclusion: **restart 33 is still the best tested value and should remain the baseline.**

---

## 15. ILU(0) with default ASM restriction remained numerically fragile

With ILU(0) and the then-default ASM restriction behavior:

- 1 rank: 1477 it, 6.90 s
- 2 ranks: 2211 it, 9.74 s
- 4 ranks: hit 5000 it, stalled at about `5.54e-6`, 17.03 s
- 8 ranks: 2276 it, 7.95 s
- 16 ranks: 2459 it, 9.40 s

Again, the 4-rank partition was pathological.

---

## 16. ILU(0) + ASM BASIC fixed the 4-rank pathology

With:

```text
factor_levels = 0
asm_type = basic
```

the sweep became:

- 1 rank: 1477 it, 6.88 s
- 2 ranks: 2206 it, 9.76 s
- 4 ranks: 1852 it, 6.42 s
- 8 ranks: 1832 it, 6.52 s
- 16 ranks: 2585 it, 10.10 s

This was decisive evidence that BASIC is the better general default for this benchmark family.

The program defaults were therefore changed to:

```text
bench_fieldsplit_u_sub_pc_factor_levels = 0
bench_fieldsplit_u_pc_asm_type = basic
```

At this point, overlap was still left at 1 as the conservative/default value.

---

## 17. ASM overlap sweep revealed a new high-performance regime

With ILU(0) and BASIC now the program defaults, overlap was swept at 4 and 8 ranks.

### 4 ranks

- overlap 0: 4714 it, setup 0.08 s, solve 6.83 s
- **overlap 1: 1852 it, setup 0.14 s, solve 6.42 s**
- overlap 2: 1887 it, setup 0.19 s, solve 8.71 s
- overlap 3: 1991 it, setup 0.22 s, solve 10.76 s

At 4 ranks, overlap 1 still wins.

### 8 ranks

- **overlap 0: 3508 it, setup 0.05 s, solve 4.24 s**
- overlap 1: 1832 it, setup 0.11 s, solve 6.54 s
- overlap 2: 2503 it, setup 0.18 s, solve 13.66 s
- overlap 3: hit 5000 it, residual ~`1.71e-9`, solve 35.05 s

The key insight was that overlap 0 greatly weakens the preconditioner but makes each iteration extremely cheap.

Approximate 8-rank cost per iteration:

- overlap 0: ~1.21 ms
- overlap 1: ~3.57 ms
- overlap 2: ~5.46 ms
- overlap 3: ~7.01 ms

Thus overlap 0 can win wall time even with almost twice as many Krylov iterations.

Overlap greater than 1 is clearly not worthwhile for this problem.

---

## 18. Overlap-0 rank sweep found the current best result

Using the current defaults (ILU(0) + BASIC) and overriding only:

```text
asm_overlap = 0
```

the following rank sweep was measured:

| MPI ranks | Iterations | Setup | Solve | Setup+solve | Approx. ms/iter | Status |
|---:|---:|---:|---:|---:|---:|---|
| 6 | 5000 | 0.06 s | 6.33 s | 6.39 s | 1.27 | failed, residual ~4.43e-9 |
| 8 | 3508 | 0.05 s | 4.21 s | 4.26 s | 1.20 | converged |
| 10 | 4314 | 0.05 s | 5.02 s | 5.07 s | 1.16 | converged |
| 12 | 4546 | 0.05 s | 5.22 s | 5.27 s | 1.15 | converged |
| 14 | 3471 | 0.05 s | 3.91 s | 3.96 s | **1.13** | converged |
| **16** | **2807** | **0.05 s** | **3.32 s** | **3.37 s** | 1.18 | **converged** |

The 14-rank case has the cheapest individual iteration, but 16 ranks wins overall because its partition converges in many fewer iterations.

The current best measured configuration is therefore approximately:

```text
MPI ranks: 16
threads per rank: 1

outer solver:
    LGMRES
    restart = 33
    augment = 3
    RIGHT preconditioning
    UNPRECONDITIONED norm
    rtol = 1e-9
    max_it = 5000

PCFIELDSPLIT:
    Schur
    LOWER
    SELFP

K/u block:
    ASM
    type = BASIC
    overlap = 0
    local KSP = PREONLY
    local PC = ILU(0)
    ordering = natural

lambda/Schur block:
    PREONLY
    Jacobi

performance:
    setup ~0.05 s
    solve ~3.32 s
    setup+solve ~3.37 s
    2807 iterations
    final relative residual ~9.96e-10
```

This is the fastest result obtained so far.

Relative to the old ~55 s SuperLU_MT run, the measured wall-time reduction is about:

$$
55 / 3.37 \approx 16.3\times.
$$

This is a wall-clock comparison, not an algorithm-identical comparison, because the two methods perform very different numerical work.

---

## 19. What the overlap-0 results mean

With PETSc-generated ASM subdomains and zero overlap, the method is conceptually very close to block Jacobi: each MPI rank approximately solves only its local owned `K` block with ILU(0), with no halo/overlap rows in the local factor.

This is why it is so cheap per iteration.

However, convergence is strongly partition dependent:

- 6 ranks failed at 5000 iterations
- 10 and 12 ranks required over 4000 iterations
- 14 and 16 ranks happened to be much better
- 16 ranks gave the best total time

Therefore overlap 0 is an excellent **performance mode**, but not a robust rank-independent mathematical preconditioner.

For general/default use, overlap 1 remains the safer choice unless performance on a known rank count is the main goal.

---

## 20. Current program defaults and recommended performance override

The user has already changed the program defaults to:

```text
ASM type = BASIC
ILU factor level = 0
```

The conservative default overlap should remain:

```text
ASM overlap = 1
```

unless the program is specifically tuned to the tested 16-rank machine/problem.

For the current large benchmark on this 16-core machine, the best measured performance command is:

```bash
BENCH_NUM_THREADS=1 BENCH_MODE=validate \
mpiexec -n 16 python petsc_mpi_hybrid_bench_1.py \
    -bench_fieldsplit_u_pc_asm_overlap 0
```

Expected result from the latest run:

```text
setup              ~0.05 s
LGMRES solve       ~3.32 s
iterations          2807
relative residual  ~9.96e-10
```

---

## 21. Main technical conclusions

### A. The biggest gains were algorithmic, not from threads

The original SuperLU_MT approach spent enormous effort on fill-heavy sparse factorization. PETSc won by accepting a weaker local approximation and doing more Krylov iterations.

A one-thread PETSc solve already beat the old 16-thread SuperLU_MT run by a large margin before serious MPI tuning began.

### B. The cheap Schur approximation is not the bottleneck

`SELFP + Jacobi` worked surprisingly well. `-log_view` showed the Schur-side solve consumed only milliseconds across more than a thousand applications.

There has been no evidence that spending more effort on the Schur block would improve total runtime enough to justify the cost.

### C. PETSc ILU(k) is dramatically better for this matrix than SciPy/SuperLU `spilu`

SciPy's `spilu` required ~43 s just to build its factor in the PETSc-like SciPy experiment, whereas PETSc ILU(1) setup was well under a second and PETSc ILU(0) setup is around 0.1 s or less in many parallel runs.

This is not merely an implementation-speed difference; the ILU algorithms themselves are different.

### D. Factor strength should be judged by wall time, not iteration count

ILU(2) had fewer iterations than ILU(1), and ILU(1) fewer than ILU(0), yet ILU(0) was much faster overall.

The dominant quantity is approximately:

$$
N_{\text{iterations}}
\times
\text{cost per preconditioner application}.
$$

### E. Natural ordering is best for the tested matrix

Orderings that reduced factor fill weakened the preconditioner enough to increase total time.

### F. LGMRES restart 33 is a genuine sweet spot for this case

Both smaller and larger restart values were worse with ILU(0). Larger Krylov spaces did not reduce iterations enough to offset added orthogonalization work.

### G. One-level domain decomposition is the main robustness limitation

MPI kernels scale reasonably well per iteration. What varies wildly is the convergence rate of the one-level local preconditioner as the rank partition changes.

This explains:

- pathological 3-rank behavior on the small problem
- pathological 4-rank behavior with RESTRICT
- 6-rank failure with overlap 0
- large non-monotonic variation in iteration counts across rank counts

### H. BASIC is a better default than RESTRICT for this benchmark

With overlap 1, BASIC removed catastrophic rank-specific convergence failures seen with RESTRICT.

### I. Zero overlap can be extremely fast but is less robust

At 16 ranks, overlap 0 gave the best measured time, but other rank counts had much worse convergence or failed.

This should be treated as a tuned performance configuration, not necessarily as the general-purpose default.

---

## 22. Remaining open experiments

The most relevant next experiments, in order of usefulness, are:

### 1. Test PETSc block Jacobi directly at 16 ranks

Because zero-overlap ASM is effectively block-Jacobi-like, test whether a direct `bjacobi` K-block PC reproduces the same behavior and perhaps reduces overhead:

```bash
BENCH_NUM_THREADS=1 BENCH_MODE=validate \
mpiexec -n 16 python petsc_mpi_hybrid_bench_1.py \
    -bench_fieldsplit_u_pc_type bjacobi
```

The existing local `sub_` options should continue to specify `PREONLY + ILU(0)`.

### 2. Compare `-log_view` for overlap 0 and overlap 1

Especially useful at 8 or 16 ranks to quantify exactly where the approximately 3x reduction in per-iteration cost comes from.

### 3. Test LGMRES augmentation dimension

Restart has been tuned, but augmentation dimension has remained at 3. A possible sweep is:

```text
augment = 0, 1, 2, 3, 5, 8
```

using ILU(0), restart 33.

This has not yet been tested.

### 4. Consider a more aggressive PETSc compiler build

The current PETSc log showed compiler flags roughly equivalent to `-g -O`, not an aggressive native build.

A separate PETSc build with something like:

```text
-O3 -march=native -mtune=native
```

may provide a modest additional improvement in the sparse kernels.

This has not yet been tested.

### 5. Two-level domain decomposition for robustness at larger scale

If the goal moves from "fastest solution on this 16-core machine" to "robust scaling as problem and rank count grow", the next architectural step should be a coarse/global correction.

`PCHPDDM` is probably the most natural first candidate to investigate given the current overlapping-domain-decomposition structure.

The current one-level ASM / block-Jacobi-like approach is fast but cannot be expected to give rank-independent convergence.

---

## 23. Important benchmarking cautions

1. Runs that hit `max_it` are not valid time-to-solution comparisons. Their per-iteration cost is still useful.
2. Some older benchmark results used `rtol=1e-8` and `max_it=2000`; later tuning uses `rtol=1e-9` and `max_it=5000`. Do not compare their iteration counts directly without noting the tolerance change.
3. With RIGHT preconditioning and UNPRECONDITIONED norm, PETSc's residual is the physical residual used for convergence.
4. The callback no longer performs an extra distributed MatMult each iteration.
5. `KSP.setUpOnBlocks()` should remain in the benchmark timing path so setup is not hidden inside solve time.
6. The current MPI tests use one software thread per rank. PETSc `SeqAIJ` local ILU does not automatically become a highly threaded BLAS workload merely because BLAS thread variables are increased.
7. The replicated SciPy/HDF5 input on every rank is acceptable for current benchmarking but not for very large distributed production problems.

---

## 24. Current best-known baseline

### Robust/general baseline

```text
1 or several MPI ranks
LGMRES restart 33, augment 3
RIGHT preconditioning
UNPRECONDITIONED norm
rtol 1e-9
max_it 5000

FIELDSPLIT Schur LOWER
SELFP

K:
    ASM BASIC
    overlap 1
    local PREONLY + ILU(0)
    natural ordering

lambda:
    PREONLY + Jacobi
```

### Fastest measured configuration on the current 16-core machine

```text
16 MPI ranks
1 thread/rank
ASM BASIC
overlap 0
ILU(0)
natural ordering
LGMRES restart 33
augment 3

setup ~0.05 s
solve ~3.32 s
total ~3.37 s
2807 iterations
final true relative residual ~9.96e-10
```

---

## 25. Short handoff paragraph for a future session

The current PETSc KKT solver uses RIGHT-preconditioned LGMRES with an UNPRECONDITIONED physical residual norm and a Schur `PCFIELDSPLIT` with LOWER factorization. The Schur preconditioner is SELFP with Jacobi and is essentially free. The `K` block is the important part. PETSc `ILU(0)` is much faster than ILU(1)/ILU(2) because repeated triangular solves dominate runtime; natural ordering and LGMRES restart 33 are best among tested values. `ASM BASIC` is much more robust than RESTRICT. The safe/default overlap is 1, but overlap 0 produces extremely cheap iterations and the fastest measured run on the 16-core machine: 16 MPI ranks, 1 thread/rank, ILU(0), BASIC, overlap 0, 2807 iterations, ~3.32 s solve and ~3.37 s setup+solve at `rtol=1e-9`. However, overlap-0 convergence is strongly partition dependent (for example, 6 ranks failed at 5000 iterations), so a genuine two-level/coarse-space method remains the likely next step for robust scaling to larger problems and rank counts.

## 26. Sample bash cmd to run the benchmarks

Use the following loop (with ``)
```bash
for N in 8 16; do \
  HWLOC_COMPONENTS=-gl \  # Suppress the message "Authorization required ..." message
  BENCH_NUM_THREADS=1 \ 
  BENCH_MODE=validate \
  mpiexec -n $N python petsc_mpi_hybrid_bench_1.py \
   -bench_fieldsplit_u_pc_asm_overlap 0; \
done
```
