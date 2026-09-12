# J2 element-formulation study protocol

The implementation and outstanding-run state was frozen before the J2 JIT
feasibility investigation, then resumed after the verified Numba acceleration
was merged.  See `FROZEN_STATE.md` for completed result fingerprints,
remaining runs, and resumption rules.

This study separates four numerical axes while keeping every unlisted input fixed. Case definitions are constructed from the checked-in square-prism or circular-bar base deck by `verification.run_j2_formulation_study`; they are not independent copies that can silently drift.

## Questions and comparisons

1. **Post-processing:** Numerical conclusions use nodal coordinates, reactions, and cell/state fields read directly from HDF5. ParaView views must use the same accepted time, warp factor, camera, and color range and are supporting evidence only.
2. **Mesh dependence:** Compare `coarse_hex8_fbar` with `refined_hex8_fbar`, and `coarse_hex8` with `refined_hex8`. Then compare the Hex8/Hex8-Fbar gap on each mesh. A locking diagnosis is supported if standard Hex8 converges slowly from the stiff side while F-bar is less mesh-sensitive.
3. **Material dependence:** Compare `soft_bulk_hex8_fbar` with `soft_bulk_hex8`, then compare that formulation gap with the reference-bulk coarse pair. A smaller gap at half bulk modulus supports an incompressibility-driven locking diagnosis.
4. **Interpolation order:** Compare `coarse_hex20` with the coarse Hex8 and Hex8-Fbar histories. Hex20 is a higher-order standard-element reference, not an assumed locking-free solution.
5. **Circular benchmark:** Complete `circular_hex8_fbar` and `refined_circular_hex8_fbar`. Establish convergence of the monitored middle-surface radial displacement before applying a tolerance to the published `-3.740 mm` final value.

For every comparison, report force--elongation, peak force and its location, middle width/radius, maximum and axial localization of equivalent plastic strain, raw and material-seen `J`, hydrostatic-stress variation, accepted increments, Newton iterations, line-search work, and cutbacks.

The circular comparison overlays middle-surface radial displacement versus
imposed elongation with the numerical literature points documented in
`REFERENCE_DATA.md`. The square-prism comparisons likewise plot middle
half-width versus imposed elongation. Reference curves are kinematic evidence,
not substitutes for the stress-distribution and mesh-sensitivity diagnostics.

## Commands

List the exact case matrix:

```bash
python -m verification.run_j2_formulation_study --list-cases
```

Run every outstanding case to the initial `t=0.5` admissibility gate in an
isolated, resumable bundle with:

```bash
conda run --no-capture-output -n py3.14 \
  python -m verification.run_j2_formulation_batch \
  --run-root examples/results/j2_formulation_study_numba \
  --phase gate --num-processes N \
  --circular-restart /absolute/path/to/restart_000002.h5
```

Repeat with `--resume` after interruption.  After inspecting the gate report,
resume every gated database to `t=1` with `--phase complete --resume`.  The
batch defaults to the verified Numba backend, constrains nested numerical
libraries to one thread per worker, retains full solver and `/usr/bin/time`
logs, and writes a provenance manifest.  `--num-processes` remains an explicit
user choice.

For an explicitly diagnosed case whose accepted increments consistently need
slightly more Newton iterations than the base deck permits for step growth,
the batch accepts a repeatable, case-local override such as
`--growth-threshold refined_hex8_fbar=6`. The resolved override is recorded in
the command, batch invocation provenance, and HDF5 input segment beginning at
the selected restart. It does not alter other study cases.

Generate all direct-HDF5 comparisons and plots with:

```bash
conda run --no-capture-output -n py3.14 \
  python -m verification.summarize_j2_formulation_study \
  --run-root examples/results/j2_formulation_study_numba \
  --expected-time 0.5
```

Use `--expected-time 1.0` for the final report.  The report generator rejects
missing or endpoint-inconsistent databases.

To run just one case outside the batch, use `--stop-time 0.5` for its initial
gate. A complete square-prism case is launched as:

```bash
/usr/bin/time -v conda run --no-capture-output -n py3.14 \
  python -m verification.run_j2_formulation_study \
  --case CASE --num-processes 2 --debug-timing --j2-backend numba
```

Omit `--j2-backend numba` to retain the interpreted reference path. Backend
choice is written to the execution section of `run_log.json` and does not
change material or restart identity.

Resume into the same result database by selecting a durable checkpoint:

```bash
/usr/bin/time -v conda run --no-capture-output -n py3.14 \
  python -m verification.run_j2_formulation_study \
  --case CASE --restart-from RELATIVE_OR_ABSOLUTE_RESTART_H5 \
  --num-processes 2 --debug-timing --j2-backend numba
```

If the result database extends beyond the selected checkpoint, resume first rolls every time-dependent dataset and the
JSON iteration history back to that exact committed time. The replaced tail is recomputed from the checkpoint.

Compare two completed square-prism databases without visualization interpolation:

```bash
python -m verification.check_j2_prism REFERENCE_RUN_H5 \
  --compare CANDIDATE_RUN_H5 --summary-only \
  --output COMPARISON_JSON --plot COMPARISON_PNG
```

Check and plot a completed circular result with:

```bash
python -m verification.check_j2_necking RUN_H5 \
  --require-reference --relative-tolerance 0.05 --plot RESPONSE_PNG
```

The 5% circular reference tolerance is only an explicit user-selectable check. It must not be adopted as an acceptance tolerance until the two circular mesh levels demonstrate adequate convergence.

## Cost warning

Two-process smoke timings on the development laptop for one increment to `t=0.02` were approximately 11 seconds for a coarse half-bulk Hex8 case, 45 seconds for coarse Hex20, and 74--87 seconds for the refined square-prism Hex8 cases. The 7680-element circular refinement is substantially larger and is not a laptop acceptance test.
