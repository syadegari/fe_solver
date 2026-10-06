# Accepted multiphase TRIP RVE results

Status: both finite-load runs and the final SVG presentation were accepted by the maintainer.
No additional numerical verification is required for this integration branch. Higher-resolution
solver/performance and visualization runs are deferred.

Closure checks: the complete Python regression suite ran 78 tests successfully, with one
existing unavailable-native-backend skip. Each archived case artifact matches the original
local file byte-for-byte; extracted histories match both retained JSON and NumPy columns,
and database time points agree exactly with the accepted-increment log. HDF5 string metadata
and retained text files were checked for machine-specific paths before staging.

## Retained artifacts

`8A56F_8x8x8/` contains `simple_shear/` and `isochoric_uniaxial/`, each with:

- The supplied TOML deck, Gmsh mesh, and immutable element-property HDF5 file.
- The original schema-3 `run.h5` and complete `run_log.json`, unchanged.
- `plots/`: approved SVGs, labeled `history.json`, compressed numerical `history.npz`,
  and a README explaining averaging, units, logarithmic strain, and stress invariants.

Git LFS stores HDF5, NumPy archives, and simulation logs. Text decks, meshes, SVGs,
and labeled report JSON remain ordinary Git files. `SHA256SUMS` records every retained
case artifact so original inputs/results and their accompanying reports can be checked.
The original local result folders are preserved separately; they are not staged.

When using a clone, retrieve the binary artifacts before reading them:

```bash
git lfs pull
cd verification/multiphase_trip_results
sha256sum -c SHA256SUMS
```

No private constitutive source, compiled library, raw Voronoi/orientation input catalog,
or machine-specific paths are included. The prepared mesh/property files suffice to
describe the discretized RVE. Postprocessing does not need the constitutive library.
The result database is an output archive, not a restart checkpoint containing complete
Gauss-point constitutive history.

## Study and accepted observations

Both cases use a periodic `[0,100]^3` domain, 512 Hex8-Fbar elements, and 729 nodes.
The initial phase element counts are 451 ferrite and 61 austenite, giving reference
volume fractions 88.0859375% and 11.9140625%. Grain orientations are runtime float64
Euler angles in the element-property file. Temperature is the compiled 300 K constant.

The duration is 2000 seconds, which matters for the rate-dependent material. Simple
shear ramps `F12` to 0.2. Isochoric tension ramps the axial stretch to 1.2 with equal
transverse stretches `1/sqrt(1.2)`. Both use exact Newton with backtracking.
The retained decks contain the actual time-controller choices: initial increment 0.1 s,
maximum 10 s, minimum `1e-5` s, and growth allowed at ten or fewer Newton iterations.

| Quantity | Simple shear | Isochoric tension |
| --- | ---: | ---: |
| Final elapsed time [s] | 2000 | 2000 |
| Accepted increments / saved states | 527 / 528 | 677 / 678 |
| Total cutbacks | 159 | 204 |
| Largest saved-state component error in `<F_raw>_V0 - Fbar` | `8.10e-15` | `6.66e-15` |
| Final mean Cauchy loading component [MPa] | `sigma12 = 382.516` | `sigma11 = 124.585` |
| Final mean stored transformation fraction in initial austenite | 0.291642 | 0.548428 |
| Final ferrite beta | 0.00307191 | 0.00399902 |
| Final austenite beta | 0.00314293 | 0.00365232 |

Stress averages use current element volume reconstructed from displacement; scalar
state averages use reference volume. All constitutive fields are the saved centroid
recoveries, not fresh material evaluations or exact quadrature-point field integrations.
The raw kinematic `F` check, separately, uses reference-volume quadrature integration
of the gradient reconstructed from nodal displacement, not the material F-bar projection.

Six-component stress plots use `[11,22,33,12,23,13]`. Shear axes are `2 hbar12` and
tension axes are `hbar11 = ln(lambdabar)`, with `hbar = log(Fbar Fbar^T)/2`.
The underlying unscaled Hencky component is retained in the numerical histories.
The invariant plot distinguishes von Mises of averaged stress from the average of
local von Mises stress; neither is asserted to be this material's yield criterion.
Beta is the model's internal parameter, not an independently defined equivalent plastic strain.

The maintainer inspected both phase blocks and evolving fields in ParaView and accepted
the results. This closes the integration/periodic workflow milestone, not an independent
experimental validation of the opaque constitutive model.

## Runtime provenance

Both original decks and the embedded resolved input specify `scipy_splu`. The actual
execution recorded in both logs used the CLI override `superlu_mt` with 16 global-solver
threads, plus 16 complete-element worker processes and one BLAS thread per worker.
Preserving this distinction avoids rewriting the supplied inputs to disguise an execution override.

The databases report Git revision `5b8582c2275707563e484ac34a2e95e0a84f1830`.
That revision belongs to the private companion repository containing the input deck;
the existing recorder obtains Git identity from the deck directory. It is **not** an
independently recorded revision of the FE source used by the remote runs. The retained
data therefore do not establish an exact historical FE source snapshot. The reporting
code and archived data are versioned together here without inventing missing provenance.
The report implementation is committed as `9d269c0`; this archive is committed directly
on top of it. The complete study is marked by the `multiphase-trip-integration-complete` tag.

Both phase definitions record the same constitutive library SHA-256 in `run.h5`:
`b556706bad2d1eddfa3275624951b832a07c6994a1a751d91beb2e06201febed`.
This identifies the original library without distributing it. Recompilation may produce
a different binary hash; matching constitutive source/constants remains necessary for
repeating the physical calculation. The logs retain timing as execution diagnostics,
not as a controlled performance comparison.

## Read results and regenerate reports

Run these from the FE repository root in `py3.14`:

```bash
TRIP_ARCHIVE=verification/multiphase_trip_results/8A56F_8x8x8
TRIP_REPORT_DIR="$(mktemp -d)"
conda run --no-capture-output -n py3.14 python -m verification.summarize_multiphase_trip \
  "$TRIP_ARCHIVE/simple_shear/run.h5" --output "$TRIP_REPORT_DIR/simple_shear"
conda run --no-capture-output -n py3.14 python -m verification.summarize_multiphase_trip \
  "$TRIP_ARCHIVE/isochoric_uniaxial/run.h5" --output "$TRIP_REPORT_DIR/isochoric_uniaxial"
conda run --no-capture-output -n py3.14 python -m fe_solver postprocess \
  "$TRIP_ARCHIVE/simple_shear/run.h5" --output "$TRIP_REPORT_DIR/simple_shear.xdmf"
conda run --no-capture-output -n py3.14 python -m fe_solver postprocess \
  "$TRIP_ARCHIVE/isochoric_uniaxial/run.h5" --output "$TRIP_REPORT_DIR/isochoric_uniaxial.xdmf"
```

Open the XDMF with ParaView's **XDMF Reader T**. The complete timeline contains separate
ferrite/austenite blocks, shared displacement/stress/strain, and the selected phase-state fields.
The reader commands deliberately write elsewhere rather than altering the retained reports.

To repeat a solve, first build the private companion's ABI-5 library and place a copy at
`build/libmultiphase_trip.so` relative to **each** archived deck. Also build the FE SuperLU_MT
bridge as described in the repository README. Then, for example:

```bash
conda run --no-capture-output -n py3.14 python -m fe_solver solve \
  "$TRIP_ARCHIVE/simple_shear/8a56f_8x8x8_simple_shear.toml" \
  --num-processes 16 --solver-backend superlu_mt --num-threads 16 --debug-timing
```

Use the uniaxial deck for the other case. These explicit concurrency values reproduce the
logged execution choices; choose lower counts on a smaller machine. New results go to the
deck's separate `results/` directory, not over the archived database. Repeating solves requires
the private model; rebuilding plots or XDMF does not. Raw-domain preprocessing instructions
remain in the companion repository's `PREPROCESSING.md`.
