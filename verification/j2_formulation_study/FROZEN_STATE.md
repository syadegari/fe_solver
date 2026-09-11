# Frozen J2 formulation-study state

Frozen on 2026-09-08 before the J2 Numba feasibility investigation.

Resumed on 2026-09-09 after the verified Numba implementation was
fast-forwarded into this branch at commit `aea6655`.  Material-point, element,
serial/process, restart, and full-solver comparisons found only roundoff-level
differences, so the outstanding production runs use the explicit `numba`
backend.  The pre-Numba databases remain valid study evidence.

Continuation note (2026-09-11): the remote completion batch finished
`soft_bulk_hex8_fbar`, `soft_bulk_hex8`, and `coarse_hex20` through `t=1`.
The `refined_hex8_fbar` case then settled at `dt=7.8125e-5`, with many
successive, first-attempt convergences taking exactly six Newton iterations.
Because the base growth threshold was five, the controller could not recover
from the earlier cutbacks. The run was stopped cleanly at `t=0.857578125` and
will recompute the tail from its latest retained material-state restart at
`t=0.8`, using the case-local completion override
`--growth-threshold refined_hex8_fbar=6`. The other cases retain the base
threshold of five. This is the smallest relaxation supported by the observed
history; a larger threshold has not been adopted.

The formulation-study implementation is on branch
`investigation/j2-formulation-study`.  The original code checkpoint preceding
this status file is commit `a14b735` (`Add J2 element formulation study
matrix`).

## Completed implementation and checks

- The nine orthogonal study cases are constructed by
  `verification.run_j2_formulation_study` from the checked-in base decks.
- Coarse and refined Hex8 meshes, the coarse serendipity Hex20 mesh, and the
  coarse and refined circular meshes are checked in.
- Direct-HDF5 square-prism and circular-bar extractors are available.
- Restarting into an existing result database transactionally truncates the
  database and JSON history to the selected committed checkpoint before
  recomputing the tail.
- All 42 unit/integration tests passed at the freeze point.
- One-increment, two-process smoke solves reached `t=0.02` for
  `soft_bulk_hex8_fbar`, `soft_bulk_hex8`, `coarse_hex20`,
  `refined_hex8_fbar`, and `refined_hex8`.

Completed focused diagnostics are retained under
`verification/material_point_results/`:

- 400-step prescribed-deformation uniaxial and simple-shear histories, with
  200-step endpoint comparisons;
- evolved material and Hex8-Fbar tangent checks at the circular-bar
  `t=0.4 -> 0.405` state;
- reduced-tangent/Newton replay of the small prism at `t=0.43 -> 0.44`, both
  without and with residual backtracking.

The first complete coarse square-prism comparison is retained under
`verification/j2_prism_results/`.  The two result databases reached `t=1`:

| Case | Accepted increments | Newton iterations | Cutbacks | Recorded wall time |
| --- | ---: | ---: | ---: | ---: |
| coarse Hex8-Fbar | 159 | 1179 | 3 | 2543.206 s |
| coarse Hex8 | 35 | 443 | 1 | 1568.997 s |

For those runs, detailed timing attributes 2476.643 s and 1543.736 s,
respectively, to element evaluation across ordinary and line-search
assemblies.  The F-bar run recorded 2257 line-search trials; the standard
Hex8 run recorded 1352.

## Existing result state

Result databases are generated artifacts and are not the authoritative study
definition.  Their state at the freeze point is:

| Study case | State on disk |
| --- | --- |
| `coarse_hex8_fbar` | complete through `t=1` |
| `coarse_hex8` | complete through `t=1` |
| `soft_bulk_hex8_fbar` | smoke result through `t=0.02` |
| `soft_bulk_hex8` | smoke result through `t=0.02` |
| `coarse_hex20` | smoke result through `t=0.02` |
| `refined_hex8_fbar` | smoke result through `t=0.02` |
| `refined_hex8` | smoke result through `t=0.02` |
| `circular_hex8_fbar` | interrupted result through `t=0.4440625`; latest durable restart is `t=0.4` |
| `refined_circular_hex8_fbar` | not run |

Fingerprints of the three substantial databases and their logs at the freeze
point are:

```text
88b80b54b856e3c64c464bcc6e548f7c7ed3a1520c1a90d664c6b9493dd3bcc3  examples/results/j2_necking_prism_small_hex8_fbar/run.h5
5e579fd2e71dbfb1f27919c2ee52b254497d2cdc3f68c3c947459fa392a6c408  examples/results/j2_necking_prism_small_hex8_fbar/run_log.json
043547b4891b50b4235eea6e087b301df0ddbd79d93508fba7f0b81063bb27d6  examples/results/j2_necking_prism_small_hex8/run.h5
5adc6c7da404f08a2d4bc4a455cb52398acc33f31d347fda8b68a446b4cf642f  examples/results/j2_necking_prism_small_hex8/run_log.json
06699a38fcce5be2b20162189b5779402f11eda59847bebcc1950d313e3193d2  examples/results/j2_necking_bar_hex8_fbar/run.h5
ebceb7e6d4fa34540398fe0797c99b1d554204bc3f8f1c16c954aff9c652ae8e  examples/results/j2_necking_bar_hex8_fbar/run_log.json
```

Resuming the circular baseline from its `t=0.4` restart will intentionally
discard and recompute the current `0.4 < t <= 0.4440625` database and log tail.

## Outstanding simulation sequence

Use the `t=0.5` gate before committing substantial resources to any complete
run.  The preferred order is:

1. Run `soft_bulk_hex8_fbar` and `soft_bulk_hex8` through `t=0.5`, inspect
   convergence, and then continue both to `t=1`.
2. Run `coarse_hex20` through `t=0.5`, inspect convergence, and then continue
   it to `t=1`.
3. Run the paired `refined_hex8_fbar` and `refined_hex8` cases through
   `t=0.5`, inspect them, and then continue both to `t=1`.
4. Resume `circular_hex8_fbar` from its durable `t=0.4` restart, first to
   `t=0.5` and then to `t=1`.
5. Run `refined_circular_hex8_fbar` first through `t=0.5` and then to `t=1`.

The preferred resumable batch command for the complete gate is:

```bash
conda run --no-capture-output -n py3.14 \
  python -m verification.run_j2_formulation_batch \
  --run-root examples/results/j2_formulation_study_numba \
  --phase gate --num-processes N \
  --circular-restart /absolute/path/to/restart_000002.h5
```

The circular seed is the frozen baseline's durable `t=0.4` restart.  The batch
uses separate result directories and therefore does not truncate the frozen
partial circular database.  Re-run an interrupted phase with `--resume` and
the same arguments; it selects the latest retained per-case restart and skips
cases that already reached the phase endpoint.

After inspecting the `t=0.5` gate report, continue the same run root with:

```bash
conda run --no-capture-output -n py3.14 \
  python -m verification.run_j2_formulation_batch \
  --run-root examples/results/j2_formulation_study_numba \
  --phase complete --num-processes N --resume
```

Both phases use one BLAS/OpenMP thread per element worker, capture verbose
`/usr/bin/time` output, tee the solver output into per-attempt logs, and update
an on-disk provenance manifest after every state change.  The exact Python
package versions used for the Linux x86_64 worker are pinned in
`requirements-j2-study-linux-x86_64.lock`.

## Outstanding comparisons and diagnostic extraction

After the simulations complete, perform these comparisons from HDF5 rather
than from interpolated visualization fields:

1. Compare the half-bulk Hex8/Hex8-Fbar gap with the completed reference-bulk
   coarse gap.  This is the material/compressibility diagnostic.
2. Compare coarse with refined response separately for Hex8 and Hex8-Fbar,
   and then compare how the formulation gap changes with refinement.  This is
   the locking/mesh-dependence diagnostic.
3. Compare Hex20 against both complete coarse Hex8 histories.  Hex20 is a
   higher-order control, not an assumed reference solution.
4. Compare coarse and refined circular histories and establish convergence of
   the monitored final radial displacement before applying a tolerance to the
   published `-3.740 mm` value.

For each square-prism comparison, report force--elongation, peak force and its
location, middle width, maximum and axial localization of equivalent plastic
strain, raw and material-seen Jacobian ranges, hydrostatic-stress variation,
accepted increments, Newton work, line-search work, cutbacks, and timing.  For
the circular pair, report the analogous force, monitored radius/displacement,
plastic localization, nonlinear work, and timing quantities.

ParaView views may be made afterward with identical accepted time, warp,
camera, and color scale.  They are supporting visual evidence only.

Generate the direct-HDF5 gate or final report and plots with:

```bash
conda run --no-capture-output -n py3.14 \
  python -m verification.summarize_j2_formulation_study \
  --run-root examples/results/j2_formulation_study_numba \
  --expected-time 0.5
```

Use `--expected-time 1.0` after the completion phase.  The summarizer rejects
missing or endpoint-inconsistent result databases.

## Resumption rule

The Numba equivalence prerequisite has been satisfied.  All new production
outputs must still use the isolated batch run root so none of the frozen
databases above are overwritten.
