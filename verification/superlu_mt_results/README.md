# SuperLU_MT production-solver comparison

Status: numerical backend comparison completed on 2026-09-28.  The separate
modified-Newton behavior review is deferred.

## Study definition

The four runs use the same `8 x 8 x 8` periodic elastic-core/J2-matrix cube,
simple shear to `F12 = 0.2`, material data, time controller, tolerances, and
backtracking controls.  The two study axes are:

| axis | values |
| --- | --- |
| nonlinear method | `newton`, `modified_newton` |
| KKT backend | `scipy_splu`, `superlu_mt --num-threads 16` |

All runs use 16 persistent element processes with one BLAS thread per worker.
The SuperLU_MT runs additionally use 16 global-solver threads.  The mesh has
729 nodes, 512 elements, 2,187 displacement unknowns, 654 multipliers, and a
`2,841 x 2,841` KKT matrix with 143,235 stored entries.  The databases report
Git revision `3d18fa2b054503339e79d3234eb2b84c6907d230` because the selectable-
backend integration was still uncommitted when the source tree was archived
for the remote runs.  No source file was changed on the remote machine.  The
integration source used by the runs is represented by commit
`74b5880eaa72770435c6a9e0a9d8a926d292a095`; this record is committed directly
on top of it.

The native bridge was configured, built, and tested with:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Denable_internal_blaslib=ON \
  -DTHREAD_API=PTHREAD \
  -DPLAT=_PTHREAD

cmake --build build -j
ctest --test-dir build --output-on-failure
```

The large HDF5 databases remain ignored under `examples/results/`.  The two
retained JSON files were generated directly from them with:

```bash
python -m verification.compare_run_databases \
  examples/results/periodic_core_j2_matrix_shear_hex8_fbar_newton_splu/run.h5 \
  examples/results/periodic_core_j2_matrix_shear_hex8_fbar_newton_superlu_mt/run.h5 \
  --rtol 1e-8 --atol 1e-8 \
  --json verification/superlu_mt_results/newton_backend_comparison.json

python -m verification.compare_run_databases \
  examples/results/periodic_core_j2_matrix_shear_hex8_fbar_modified-newton_splu/run.h5 \
  examples/results/periodic_core_j2_matrix_shear_hex8_fbar_modified-newton_superlu_mt/run.h5 \
  --rtol 1e-8 --atol 1e-8 \
  --json verification/superlu_mt_results/modified_newton_backend_comparison.json
```

The input is the production periodic-composite shear deck with the mesh changed
to `heterogeneous_periodic_cube_8x8x8.msh`, `grow_if_newton_iterations_le = 8`,
and the result directory isolated per run.  Only `nonlinear.method` changes
between the Newton cases.  Backend selection and the SuperLU_MT thread count
are command-line execution choices.

## Backend agreement

Both backend pairs have identical database structure and accepted-time arrays
and pass the complete HDF5 comparison at `rtol = atol = 1e-8`.

| maximum difference | Newton pair | modified-Newton pair |
| --- | ---: | ---: |
| matrix Cauchy stress, absolute [MPa] | `7.507e-10` | `7.013e-10` |
| matrix Cauchy stress / field maximum | `1.765e-12` | `2.699e-12` |
| nodal displacement, absolute | `1.249e-15` | `2.277e-16` |
| nodal displacement / field maximum | `6.102e-15` | `7.028e-14` |
| constraint reaction, absolute | `2.162e-12` | `2.726e-12` |
| constraint reaction / field maximum | `2.999e-13` | `6.716e-13` |
| equivalent plastic strain, absolute | `4.607e-15` | `8.174e-16` |
| equivalent plastic strain / field maximum | `1.946e-14` | `1.447e-9` |

The comparator's largest pointwise relative errors occur where both compared
values are essentially zero; they are not representative field errors.  The
field-maximum-scaled values above avoid that near-zero denominator artifact.
At the common accepted state `t = 0.01`, exact and modified Newton also agree:
their stress, displacement, and reaction differences are about `1e-12` of
their field maxima, and their plastic states are exactly equal.

## Secondary timing observations and nonlinear paths

The purpose of these four runs is numerical correctness of the production
backend integration.  The timing data are retained as diagnostics, not as a
general performance or scaling result.  This `8 x 8 x 8` problem has a small
`2,841 x 2,841` KKT system; one SciPy factorization takes only about 0.24 seconds
on this machine.  Startup, assembly, line search, triangular solve, hardware,
linked BLAS, ordering, thread count, and nonlinear factorization frequency can
therefore materially change both the measured factorization ratio and the
end-to-end ratio.

| quantity [s] | Newton SciPy | Newton MT | modified SciPy | modified MT |
| --- | ---: | ---: | ---: | ---: |
| elapsed wall time | `152.435` | `89.531` | `65.234` | `60.908` |
| KKT factorization | `83.429` | `18.448` | `6.173` | `1.327` |
| KKT triangular solve | `0.603` | `1.029` | `1.200` | `1.589` |
| primary assembly | `8.893` | `9.086` | `3.857` | `3.736` |
| line-search assembly | `51.640` | `53.256` | `48.336` | `48.543` |

For this exact-Newton run, the observed ratios are `4.522x` for factorization,
`4.315x` for factorization plus triangular solves, and `1.703x` end to end.
Both runs complete 61 increments without cutback and use 351 Newton corrections.
Their accepted increment histories are exactly equal.  Small roundoff
differences alter a few late line-search decisions: SciPy records five
backtracks and SuperLU_MT four, without changing the accepted states materially.

For this modified-Newton run, the observed factorization ratio is `4.650x`, but
the end-to-end ratio is only `1.071x`.  Modified Newton performs only 25
factorizations while spending about 48.4 seconds in 496 line-search trial
assemblies, so sparse factorization is no longer the dominant cost.  The
SuperLU_MT triangular solve is slower for this small KKT system in both
nonlinear methods; factorization savings dominate it in the exact-Newton run.

As separate sizing context, a prior `16 x 16 x 16` case on the same remote
machine took roughly 25--27 seconds per SciPy factorization and 3.4 seconds with
16-thread SuperLU_MT.  That observation suggests why the native backend matters
more for larger systems, but it is not part of this controlled four-run study
and is not a cross-size scaling curve.  No general speedup is inferred from
either observation alone.

## Deferred modified-Newton observation

Both modified-Newton runs follow exactly the same accepted-increment history:
10 accepted increments, 25 attempts, 15 cutbacks, and a last durable state at
`t = 0.01619870625436306`.  Both then require a cutback below `dt_min = 1e-6`
and terminate.  Their last saved periodic-average deformation-gradient errors
are `6.439e-15` (SciPy) and `6.883e-15` (SuperLU_MT).

This endpoint is mechanically significant: it gives prescribed
`F12 = 0.0032397413` and macroscopic `P12 = 259.803 MPa`, while the matrix's
initial J2 shear yield stress is `450/sqrt(3) = 259.808 MPa`.  The difficulty
therefore begins precisely as the matrix crosses its elastic-plastic transition.

The durable log shows full-step line-search acceptance (`alpha = 1`) followed
by slow, approximately linear force-residual reduction with the frozen tangent;
failed attempts reach `nonlinear.max_iterations = 30`.  This is consistent with
modified Newton losing convergence rate near the elastic-plastic transition and
does not by itself demonstrate a backend error or a modified-Newton coding bug.
Unlike exact Newton, modified Newton is not generally guaranteed to be more
robust; it exchanges tangent refactorization for linear convergence.  A review
of modified Newton, its tangent-freezing point, convergence controls, and its
interaction with backtracking is intentionally deferred to the planned broader
solver audit.

## Conclusion

The completed exact-Newton run and the common converged prefix of the
modified-Newton run independently show that `scipy_splu` and `superlu_mt`
produce the same FE results well inside the requested `1e-8` tolerance.  The
study therefore closes the production numerical-equivalence requirement for
the selectable SuperLU_MT backend.  The recorded timing is secondary evidence
only; performance characterization requires larger systems and a dedicated
thread/scaling study.
