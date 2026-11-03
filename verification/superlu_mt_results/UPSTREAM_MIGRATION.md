# SuperLU_MT fork-to-upstream migration check

Status: comparison completed on 2026-11-02, reproducer and decision record added on 2026-11-03.  Read together with the
[production-solver comparison](README.md), which compared the SciPy and
SuperLU_MT backends against each other.  This record compares two builds of
the SuperLU_MT backend against each other.

## Question

The native bridge used to build against a fork of SuperLU_MT
(`syadegari/superlu_mt`, branch `fe-integration`, commit `d503e498`) that
carried two fixes: initialization of `fact` and `trans` in `p?gstrf_init()`
and CMake support for the bundled CBLAS.  Both fixes were merged upstream as
PRs #33 and #34.  The root `CMakeLists.txt` now fetches
`xiaoyeli/superlu_mt` at commit `f2e50bfcbb7b20ef304c84cf92c038c283019863`
(tag `v4.1.0`), which additionally contains PR #35 (PLAT derived from
THREAD_API), PR #36 (per-thread workspace fix for user-supplied `work[]`),
and PR #37 (L/U storage estimates settable through `superlumt_options_t`).

The question answered here: does switching the pinned library change the
numerical response of the FE solver?

Expected answer from reading the diffs: no, bit for bit.  The bridge passes
`lwork = 0`, so every statement changed by PR #36 sits in a branch it never
executes, and PR #37 only replaces three `sp_ienv()` reads with option fields
that `pdgstrf_init()` sets to zero, which falls back to the same `sp_ienv()`
values.

## Builds

Both bridges were built from the same `native/superlu_mt.c` with identical
flags (`Release`, `enable_internal_blaslib=ON`, `THREAD_API=PTHREAD`), the
fork build via `FETCHCONTENT_SOURCE_DIR_SUPERLU_MT` pointing at a checkout of
`d503e498` and with `PLAT=_PTHREAD` on the command line, the upstream build
from the repository `CMakeLists.txt` as committed.  All 48 upstream CTest
cases pass on the upstream build.

## FE comparison: 4 x 4 x 4 periodic composite

Deck: `examples/periodic_core_j2_matrix_shear_hex8_fbar.toml` with the mesh
changed to `heterogeneous_periodic_cube_4x4x4.msh` (125 nodes, 8 Hex8 core
elements, 56 Hex8-Fbar J2 matrix elements) and the result directory isolated
per run; everything else unchanged (simple shear to `F12 = 0.2`, Newton with
backtracking, adaptive stepping).  KKT system `561 x 561` (375 displacement
and 186 multiplier unknowns) with 20,511 stored entries.  All runs use four
element processes with one BLAS thread each.  Databases were compared with
`verification.compare_run_databases`; "exact" means every one of the 19
numeric datasets is bitwise equal.

### Single solver thread

| pair | exact datasets | max difference |
| --- | ---: | ---: |
| fork run A vs fork run B (control) | 19 / 19 | 0 |
| fork vs upstream | 19 / 19 | 0 |
| upstream run A vs upstream run B (control) | 19 / 19 | 0 |

With one SuperLU_MT thread the fork and the upstream pin produce identical
databases, and each build is identical with itself.  This is the direct
answer to the migration question.

### Eight solver threads

Multithreaded SuperLU_MT is not run-to-run reproducible: with the same binary
and the same inputs, every factorization with four or more threads gives a
slightly different solution (see the KKT snapshot section below).  The
fork-versus-upstream difference at eight threads must therefore be read
against the same-binary repeat difference.  Final state at `t = 1`, maximum
absolute difference divided by the field maximum of the single-thread
reference:

| pair | displacement | reaction | matrix Cauchy stress | eq. plastic strain |
| --- | ---: | ---: | ---: | ---: |
| fork A vs fork B (control) | `3.2e-15` | `1.8e-13` | `1.1e-12` | `7.4e-15` |
| fork A vs upstream A | `6.1e-15` | `2.0e-13` | `9.9e-13` | `1.2e-14` |
| single-thread vs SciPy `splu` | `4.6e-15` | `2.0e-13` | `8.6e-13` | `7.8e-15` |

Thirteen fork and twelve upstream runs were made at eight threads.  All runs
that followed the reference increment sequence (61 increments, 399 Newton
iterations) agree with the single-thread reference to at most `5e-15`
(displacement) and `8e-13` (stress) relative.  Four fork runs and no upstream
run departed from that sequence; see the next section.

## Sporadic wrong solves with four or more threads

Two fork runs at eight threads needed one extra Newton iteration, and two
others exhausted the line search (14 backtracks), triggered one step cutback,
took 62 increments, and ended at `t = 1` with fields `1e-5` to `4e-5` relative
away from the reference.  In each case the Newton step after one specific
linear solve left a constraint residual of `8e-5`, `3e-9`, or similar where
every other iteration shows `1e-15`.  The constraint rows are linear and
exact in the KKT system, so a correct solve cannot produce that: those
particular factorizations returned wrong solutions.

To separate this from the migration, all 338 KKT systems of a deterministic
SciPy-backend run of the same case were archived and replayed through both
bridges with the original right-hand sides.  A solve is counted as wrong when
`|Ax - b|` exceeds `1e-10` relative to `max(|A 1|, |b|)`; good solves sit at
`1e-13`.

| threads | factorizations per build | fork wrong | upstream wrong | worst relative residual |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,690 | 0 | 0 | `3.4e-13` |
| 2 | 3,380 | 0 | 0 | `2.6e-13` |
| 4 | 8,450 | 1 | 0 | `2.1e-1` |
| 8 | 8,450 | 3 | 5 | `1.1e+0` |
| 16 | 3,380 | 7 | 4 | `8.7e-1` |

The rate grows with the thread count and is the same for both builds within
the sample sizes.  Thirty repeated eight-thread factorizations of the archived
`17,193`-unknown snapshot showed no failure on either build, so the rate on
production-size systems is unknown but at least much lower.  `pdgstrf` reports
`info = 0` for the wrong solves.

### Standalone reproducer

The defect reproduces outside the FE solver and outside Python.  A C program
that uses only the public SuperLU_MT API (`get_perm_c`, `pdgstrf_init`,
`pdgstrf`, `dgstrs`, the sequence of `pdgssv()`), linked against the plain
library build, on one archived `561 x 561` matrix with `b = A x_exact`:

| threads | wrong of 5,000, upstream `v4.1.0` | wrong of 5,000, fork `d503e498` |
| ---: | ---: | ---: |
| 1 | 0 | |
| 2 | 0 | |
| 4 | 1 | |
| 8 | 2 | |
| 16 | 10, 15, 6 | 8 |

Memory expansion is not involved (the library reports zero expansions), the
`v4.1.0` storage-estimate options do not change the rate, and the fix in PR
#36 is not reached because no user work array is passed.  A ThreadSanitizer
build reports more than 200 distinct races inside the factorization, the most
frequent between the pivot-row interchange in `pdgstrf_pivotL.c` and the dense
update kernels of other threads; SuperLU_MT relies partly on flag-based
synchronization the sanitizer cannot see, so these are hints rather than a
localization.  The program, the matrix, the exact numbers, and the sanitizer
summary are in [`native/superlu_mt_race/`](../../native/superlu_mt_race/README.md),
prepared so that it can be handed to the upstream maintainers without
further context.

### Decision: leave the fix upstream, keep the multithreaded backend

The fix belongs in SuperLU_MT and is left to its maintainers, with the
reproducer as the hand-off.  The library's parallel factorization is a
left-looking scheme with hand-written synchronization across a dozen source
files; a correct fix requires understanding which flag or counter is read
before its write is published, and a local patch that merely narrows the race
would be worse than none because it would turn a reproducible defect into an
irreproducible one.  The earlier upstream fix for the user-workspace race (PR
#36) was made by the maintainers from a CTest failure report alone, so a
small, self-contained reproducer is the most effective contribution from this
side.  This project also does not want to return to carrying a patched fork,
which is what this migration removed.

The multithreaded backend stays available because the FE solver already
contains the check that makes a wrong linear solve harmless:

- Every Newton iteration recomputes the force and constraint residuals from
  the trial displacement, not from the linear-solver output.  A wrong
  direction cannot be accepted as converged, since acceptance requires both
  residuals below tolerance.
- The backtracking line search rejects a direction that does not decrease the
  residual merit, and the step controller cuts the increment back when the
  line search is exhausted.  Both happened in the observed runs; the solver
  then continued from the committed state and converged.
- The cost of a wrong solve is one wasted iteration or, rarely, a cutback.
  The runs that took a cutback followed a different load path and differ from
  the reference by `1e-5` relative in the final fields, which is within the
  path dependence already accepted for adaptive stepping and well inside the
  convergence tolerances of the deck.
- With one or two threads the backend is deterministic and bit-identical with
  the fork, so a reproducible configuration remains available when needed.
- The failure was never observed on the `17,193`-unknown system, and the
  backend exists for large systems; on the small systems where it occurs the
  serial SciPy backend is the natural choice anyway.

Two follow-ups are recorded here and not done: a residual check after each
SuperLU_MT solve in the bridge (one sparse matrix-vector product, negligible
against a factorization) with refactor-and-retry, which would remove even the
wasted iteration; and re-running the reproducer when a new SuperLU_MT release
appears, before bumping the pin.

## Archived KKT snapshots

`native/kkt_debug_1.h5`, group `t_n=0.02/iteration=2` (`17,193` unknowns,
1,068,651 stored entries), factored and solved by both bridges with the
solver's defaults (COLAMD, pivot threshold 1, default panel size and relax):

| threads | fork repeat identical | fork vs upstream identical | max abs fork vs upstream | max abs upstream vs SciPy | factor time fork / upstream [s] |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | yes | yes | 0 | `7.8e-12` | 32.9 / 32.7 |
| 2 | yes | yes | 0 | `7.1e-12` | 17.0 / 16.6 |
| 4 | no | no | `6.8e-12` | `6.5e-12` | 8.9 / 9.1 |
| 8 | no | no | `8.7e-12` | `9.0e-12` | 5.4 / 5.3 |
| 16 | no | no | `6.3e-12` | `7.2e-12` | 4.5 / 4.0 |

Residuals `|Ax - b|_inf` were `4e-15` to `5e-15` in every case.  One and two
threads are deterministic and identical between the builds; from four threads
on, the fork-versus-upstream difference equals the same-build repeat
difference and the SuperLU_MT-versus-SciPy difference.

`native/kkt_debug_2.h5` (`52,281` unknowns, 3,522,771 stored entries) could
not be factored by either build with the default storage estimates: SuperLU_MT
aborts with "Storage for U columns exceeded; Need at least 176140446" from
`sp_ienv(7)`.  SciPy `splu` factors it in about 400 s.  This is the case for
exposing the PR #37 storage options through the bridge; it is unrelated to
the migration.

## Conclusion

The migration from the fork to upstream `v4.1.0` is numerically neutral:
bit-identical with one or two solver threads, and indistinguishable from
same-binary repeat noise with more.  The sporadic wrong solves at four or more
threads on small systems exist equally in both builds, are detected and
absorbed by the Newton loop, and are handed to upstream through the standalone
reproducer in `native/superlu_mt_race/`.

## Recipe

Both builds from the repository root, with `FORK` a checkout of
`syadegari/superlu_mt` at `d503e498`:

```bash
cmake -S . -B build_fork -DCMAKE_BUILD_TYPE=Release -Denable_internal_blaslib=ON \
  -DTHREAD_API=PTHREAD -DPLAT=_PTHREAD -DFETCHCONTENT_SOURCE_DIR_SUPERLU_MT="$FORK"
cmake -S . -B build_upstream -DCMAKE_BUILD_TYPE=Release -Denable_internal_blaslib=ON \
  -DTHREAD_API=PTHREAD
cmake --build build_fork --parallel && cmake --build build_upstream --parallel
```

The solver loads `build/native/libsuperlu_mt_bridge.so`, so the two bridges
were copied into place in turn.  Each FE run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m fe_solver solve DECK \
  --solver-backend superlu_mt --num-threads N --num-processes 4
python -m verification.compare_run_databases REF/run.h5 CAND/run.h5 --rtol 1e-8 --atol 1e-8
```

The KKT replay patched `fe_solver.solver._factor_kkt` to archive every KKT
matrix and first right-hand side of a SciPy-backend run, then called
`native.superlu_mt_backend.SuperLUMTFactor.factor` and `solve` on each
archived system repeatedly through each bridge and checked `|Ax - b|`.
