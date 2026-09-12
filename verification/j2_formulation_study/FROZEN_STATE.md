# Closed J2 formulation-study state

The study was closed on 2026-09-12. All production cases reached `t=1`, the
direct-HDF5 comparison report was generated, and the final test suite passed
45 tests. The generated databases, logs, and plots remain ignored artifacts;
the case definitions, extraction tools, provenance rules, and reproduction
commands in this directory are the tracked study record.

Frozen on 2026-09-08 before the J2 Numba feasibility investigation.

Resumed on 2026-09-09 after the verified Numba implementation was
fast-forwarded into this branch at commit `aea6655`.  Material-point, element,
serial/process, restart, and full-solver comparisons found only roundoff-level
differences, so the outstanding production runs use the explicit `numba`
backend.  The pre-Numba databases remain valid study evidence.

Continuation note (2026-09-11--12): the remote completion batch finished
`soft_bulk_hex8_fbar`, `soft_bulk_hex8`, and `coarse_hex20` through `t=1`.
The first `refined_hex8_fbar` completion then settled at `dt=7.8125e-5`, with
many successive, first-attempt convergences taking exactly six Newton
iterations. Because the base growth threshold was five, the controller could
not recover from the earlier cutbacks. That partial result was stopped at
`t=0.857578125` and retained only as an untracked diagnostic. The final
`refined_hex8_fbar` result was recomputed from `t=0` with the explicitly chosen
threshold eight.

The four manually launched completions all used threshold eight, as confirmed
by their resolved HDF5 input and `/usr/bin/time` command records. The refined
standard-Hex8 result retains its threshold-five prefix through `t=0.5` and
uses threshold eight afterward; both circular results and the final refined
Hex8-Fbar result use threshold eight from `t=0`. The three cases completed by
the batch retain threshold five. Consequently, final mesh-refinement
comparisons must disclose this controller difference and check its sensitivity
rather than claiming that every non-formulation input is identical.

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
- All 42 unit/integration tests passed at the original freeze point; all 45
  tests passed after the completed-run and reference-data work.
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

## Historical result state at the original freeze

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

## Completed simulation sequence

The production study used the `t=0.5` gate before committing substantial
resources to the complete runs. The sequence was:

1. `soft_bulk_hex8_fbar`, `soft_bulk_hex8`, and `coarse_hex20` passed the gate
   and were completed by the remote batch.
2. `refined_hex8_fbar`, `refined_hex8`, `circular_hex8_fbar`, and
   `refined_circular_hex8_fbar` passed the gate and were completed manually
   after the batch workflow became less convenient than direct launches.
3. The seven supplemental production databases were transferred back and
   checksum-verified. Together with the two original coarse-prism databases,
   they complete the nine-case matrix.

The resumable batch command retained for reproduction is:

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

After inspecting the `t=0.5` gate report, a reproduction can continue the same
run root with:

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

## Completed comparisons and diagnostic extraction

The final `report_t1` bundle was generated directly from HDF5 and contains
these comparisons:

1. Half-bulk and reference-bulk Hex8/Hex8-Fbar formulation gaps.
2. Coarse/refined response for Hex8 and Hex8-Fbar, including the change in the
   formulation gap with refinement.
3. Hex20 against both complete coarse Hex8 histories as a higher-order
   control, not an assumed exact solution.
4. Coarse/refined circular response, including monitored radial displacement
   and the published numerical-reference curves.

For each square-prism comparison, report force--elongation, peak force and its
location, middle width, maximum and axial localization of equivalent plastic
strain, raw and material-seen Jacobian ranges, hydrostatic-stress variation,
accepted increments, Newton work, line-search work, cutbacks, and timing.  For
the circular pair, report the analogous force, monitored radius/displacement,
plastic localization, nonlinear work, and timing quantities.

ParaView views may be made afterward with identical accepted time, warp,
camera, and color scale.  They are supporting visual evidence only.

Regenerate the direct-HDF5 gate or final report and plots with:

```bash
conda run --no-capture-output -n py3.14 \
  python -m verification.summarize_j2_formulation_study \
  --run-root examples/results/j2_formulation_study_numba \
  --expected-time 0.5
```

Use `--expected-time 1.0` after the completion phase.  The summarizer rejects
missing or endpoint-inconsistent result databases.

## Localization observation and limitation

The circular-bar solution correctly places the largest plastic deformation in
the first axial element layer next to the center symmetry plane, where the
geometric imperfection makes the radius smallest. This location is expected
for the necking instability and agrees qualitatively with the published
deformed meshes and contours.

The *width* and peak value of that localization must not, however, be treated
as mesh-objective predictions. At the final state, the coarse mesh (24 axial
layers) has maximum equivalent plastic strain `1.5152` in its first layer,
while the refined mesh (48 layers) has `1.8354` in its first, half-thickness
layer. The corresponding first-layer means are `1.4885` and `1.7806`. Thus the
refined first-to-second-layer mean-strain ratio grows from `1.037` at `t=0.7`
to `1.429` at `t=1`, confirming that the visible one-layer concentration
develops primarily in the final third. The global force/displacement and
neck-displacement agreement can therefore be good while a local plastic-strain
measure remains mesh-sensitive. Elguedj and Hughes make the same distinction
between global curves and local stress/strain fields and, when extending this
benchmark to 9 mm, explicitly describe deformation as too localized in the
first layer for an accurate solution:

<https://www.ices.utexas.edu/media/reports/2011/1135.pdf>

This is not evidence of an implementation bug. It is also not, by itself, a
proof of loss of ellipticity or pathological mesh dependence: the benchmark
has geometric necking and positive material hardening. It is evidence that the
present local Cauchy-continuum J2 model supplies no independent material length
with which to make the late-stage localization width objective. Convergence of
global response therefore does not establish convergence of peak plastic
strain, localization width, or element distortion.

For a possible distant extension with the smallest conceptual departure from
the current displacement/J2 framework, the preferred option is a scalar
implicit-gradient enhancement of an accumulated-plastic-strain or hardening
variable. It introduces a material length through a Helmholtz-type equation
and can use ordinary `C0` interpolation, but it still requires one additional
global scalar field, boundary conditions, coupled residual/tangent blocks, and
cross-element communication. It is substantially less disruptive than a
Cosserat continuum while being a genuine spatial regularization. Relevant
starting points are:

- Ramaswamy and Aravas, gradient plasticity with a von Mises example:
  <https://doi.org/10.1016/S0045-7825(98)00028-0>
- Engelen et al., implicit gradient-enhanced elastoplasticity:
  <https://doi.org/10.1016/S0749-6419(01)00042-0>

An integral nonlocal average of the same scalar variable is the next closest
alternative. It avoids an extra nodal field but requires neighborhood search,
parallel data exchange, boundary corrections, and a nonlocal consistent
tangent. A purely local viscoplastic term would be easier to add and may smooth
the computation, but it introduces loading-rate dependence and is not a
reliable guarantee of spatial mesh objectivity. F-bar, higher-order elements,
adaptive remeshing, and arc-length control address locking, approximation and
distortion, or equilibrium-path tracing; none by itself supplies the missing
localization length. Crack-band scaling becomes a low-cost option only if a
calibrated damage or strain-softening energy law is introduced later.

These observations close the present study; none of the regularization options
is part of the current implementation scope.

## Historical resumption rule

The Numba equivalence prerequisite was satisfied before the production runs.
Any future reproduction should use an isolated batch run root so historical
databases are not overwritten.
