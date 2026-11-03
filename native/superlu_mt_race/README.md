# Reproducer: sporadic wrong solutions from multithreaded SuperLU_MT

`pdgstrf()` followed by `dgstrs()` occasionally returns a solution with a
residual `|Ax - b| / |b|` between `1e-7` and `10` while reporting `info = 0`,
when run with four or more threads on the small unsymmetric saddle-point matrix
supplied here.  With one or two threads the same calls are deterministic and
accurate to `3e-15`.  The behavior is the same in SuperLU_MT `v4.1.0`
(`f2e50bfc`) and in the earlier commit `d503e498` that this project used
before, so it is not caused by any change between the two.

This directory is self-contained so that it can be handed to the SuperLU_MT
maintainers as is: a C program that uses only the public SuperLU_MT API, one
matrix in Matrix Market format, and the observations below.  Nothing here
depends on the FE solver.  How the defect was found and why the FE solver
tolerates it is recorded in
[`verification/superlu_mt_results/UPSTREAM_MIGRATION.md`](../../verification/superlu_mt_results/UPSTREAM_MIGRATION.md).

## Files

| file | content |
| --- | --- |
| `superlu_mt_race_repro.c` | reads a Matrix Market matrix from stdin, sets `b = A x_exact` for a fixed pseudo-random `x_exact`, then repeats `get_perm_c` (COLAMD), `pdgstrf_init`, `pdgstrf`, `dgstrs` with `NPROCS` threads and counts solves whose relative residual exceeds a tolerance (default `1e-8`).  The call sequence is that of `pdgssv()`, written out so that the L/U storage estimates can be set. |
| `kkt_4x4x4_0017.mtx.gz` | `561 x 561` real unsymmetric matrix with 20,511 stored entries: the KKT system `[K, C^T; C, 0]` of a small finite-element problem, with a `375 x 375` stiffness block `K`, 186 affine-constraint rows `C`, and a zero diagonal block. |

## Build and run

From the repository root, after the normal `cmake -S . -B build` configure
step, the program is an optional CMake target that links the fetched library:

```bash
cmake --build build --target superlu_mt_race_repro
zcat native/superlu_mt_race/kkt_4x4x4_0017.mtx.gz | build/native/superlu_mt_race_repro 16 5000
```

Against an arbitrary CMake build of SuperLU_MT (source tree `SRC_DIR`, build
tree `BUILD_DIR`, pthread flavor with the bundled CBLAS):

```bash
cc -O2 -o superlu_mt_race_repro superlu_mt_race_repro.c \
   -I "$SRC_DIR/SRC" -D__PTHREAD \
   "$BUILD_DIR/SRC/libsuperlu_mt_PTHREAD.a" "$BUILD_DIR/CBLAS/libblas.a" -lpthread -lm
zcat kkt_4x4x4_0017.mtx.gz | ./superlu_mt_race_repro NPROCS REPS [FILL] [TOL]
```

`FILL` is passed to `fill_lusup`, `fill_ucol`, and `fill_lsub` of
`superlumt_options_t` (SuperLU_MT `v4.1.0` and later); `0` keeps the
`sp_ienv(6..8)` defaults.  The exit status is 1 when any solve was wrong.

Typical output at 16 threads:

```
n = 561, nnz = 20511, nprocs = 16, reps = 5000, fill = 0, tol = 1e-08
nnz(L) = 43509, nnz(U) = 80171, L\U storage 1.48 MB, expansions during factorization = 0
rep 1733: WRONG SOLUTION  |Ax-b|/|b| = 4.209e+00  max|x-x_exact| = 2.793e+05  (info = 0)
...
summary: nprocs = 16, 10/5000 solves wrong (rel. residual > 1e-08), 0 with info != 0, worst rel. residual = 4.209e+00
```

## Observations

5,000 repetitions per row, library defaults for ordering (COLAMD), pivot
threshold (1.0), panel size, relaxation, and storage estimates:

| threads | wrong solves, `v4.1.0` | wrong solves, `d503e498` | worst relative residual |
| ---: | ---: | ---: | ---: |
| 1 | 0 | | `1.7e-15` |
| 2 | 0 | | `2.8e-15` |
| 4 | 1 | | `1.1e-4` |
| 8 | 2 | | `1.2e-5` |
| 16 | 10, 15, 6 (three runs) | 8 | `34` |

- The failing repetitions are different in every run, and the wrong solutions
  differ from each other.  `pdgstrf` returns `info = 0` in every case.
- The factorization itself is not reproducible at 16 threads: the first
  repetition reported `nnz(U) = 80171` in one run and `80207` in another, so
  the pivot sequence already depends on thread timing even when the solution
  is accurate.  At one and two threads `nnz(U)` is the same in every run.
- Oversubscription (32 or 64 threads on 16 cores) and running four instances
  concurrently did not raise the rate noticeably; 40-repetition runs usually
  show nothing, which is why the counts above use 5,000.
- A second matrix from the same sequence of finite-element Newton iterations
  (same sparsity pattern, different values) behaves the same way (9 of 5,000 at
  16 threads).  In a sweep over 338 such matrices, wrong solves were spread over
  all of them rather than concentrated on particular values.
- Not the cause: the initial L/U storage is sufficient (`expansions = 0`), and
  raising the estimates to 300 times `nnz(A)` with `FILL = -300` left the
  rate unchanged (6 of 5,000 at 16 threads).  No user work array is passed
  (`lwork = 0`), so the `v4.1.0` fix for user-supplied workspace does not
  apply.
- The rate does not fall with system size.  A `17,193`-unknown matrix of
  the same structure (`nnz(A)` 1,068,651; `nnz(L) + nnz(U)` 73 million, no
  expansions) gave 1 wrong solve in 600 repetitions at 16 threads, relative
  residual `0.12`, with `nnz(U)` again differing between runs (42,851,615
  versus 42,851,622).  That is the same order as the `2e-3` per
  factorization measured on the small matrix.  The small matrix is supplied
  because it reproduces in seconds rather than hours.

## ThreadSanitizer

A double-precision pthread build of `v4.1.0` with
`-fsanitize=thread -fno-omit-frame-pointer -g` (CBLAS, examples and tests off)
and the reproducer compiled the same way reports more than 200 distinct data
races in a 20-repetition run at eight threads.  On Linux 6.8 the sanitizer
aborts with "unexpected memory mapping" unless address-space randomization is
disabled for the process:

```bash
TSAN_OPTIONS="halt_on_error=0" setarch "$(uname -m)" -R ./repro_tsan 8 20 < kkt_4x4x4_0017.mtx
```

The instrumented program produced no wrong solve in those 20 repetitions,
as the sanitizer slows and serializes the threads.  SuperLU_MT synchronizes its threads partly through flags and spin waits the
sanitizer cannot see, so a share of the reports is expected to be benign.  The
most frequent pairs, as hints for where to look:

| writes in | racing reads or writes in |
| --- | --- |
| `pdgstrf_pivotL.c:156-157` (row-subscript and value interchange of the pivot row within a supernode) | `dmyblas2.c` dense kernels (`dlsolve`, `dmatvec`) called from `pdgstrf_bmod1D.c` and `pdgstrf_panel_bmod.c`; `pdgstrf_bmod1D.c:188` |
| `pdgstrf_column_bmod.c:278` (gather of the sparse accumulator into `lusup`) | `pdgstrf_bmod1D.c:142` |
| `pdgstrf_pivotL.c:140-141` (`perm_r`, `inv_perm_r` updates) | `pdgstrf_panel_dfs.c:148`, `pdgstrf_column_dfs.c:109`, `pdgstrf_panel_bmod.c:348` |

All racing locations are inside the L-factor value and subscript arrays
allocated by `pdgstrf_MemInit` / `pdgstrf_expand` at the start of the
factorization.

## Environment

AMD Ryzen 9 5950X (16 cores, 32 threads), Ubuntu 24.04.5, Linux 6.8.0, GCC
13.3.0, CMake 3.28.3, SuperLU_MT built with `-DCMAKE_BUILD_TYPE=Release
-Denable_internal_blaslib=ON -DTHREAD_API=PTHREAD`, 32-bit `int_t`.
