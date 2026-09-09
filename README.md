# Finite-strain FE prototype

This repository implements the v1 contract in `docs/IMPLEMENTATION_SPEC.md`:

- full-integration Hex8 and Hex20 serendipity elements;
- centroidal F-bar Hex8 with its complete projection tangent;
- compressible neo-Hookean material through the endpoint `F_n`/`F_np1` material-point API;
- a stateful multiplicative finite-strain J2/Voce plug-in with an analytic algorithmic tangent;
- sparse COO/CSR assembly and a whole-system sparse-LU KKT solve for affine constraints;
- exact and modified Newton, optional residual-based backtracking, recoverable trial rollback, adaptive cutback, and mandatory events;
- Gmsh Physical Groups and translational periodic node maps;
- append-only accepted-state HDF5 output, HDF5 restart, and separate temporal XDMF postprocessing.

The production element tangents use the specification's current-configuration Truesdell push-forward, geometric stiffness, and F-bar projection correction. An independent total-reference F-bar kernel and centered directional finite differences cross-check that decomposition.

## Run

The checked-in acceptance meshes can be regenerated with:

```bash
python examples/generate_elongated_block_gmsh.py --element hex8 --out examples/elongated_hex8.msh
python examples/generate_elongated_block_gmsh.py --element hex20 --out examples/elongated_hex20.msh
python examples/generate_periodic_cube_gmsh.py --out examples/periodic_cube_8x8x8.msh
python examples/generate_heterogeneous_periodic_cube_gmsh.py --out examples/heterogeneous_periodic_cube_8x8x8.msh
python examples/generate_necking_bar_gmsh.py --out examples/necking_bar_quarter_hex8.msh
python examples/generate_necking_bar_gmsh.py --core-divisions 8 --radial-divisions 6 --nz 48 \
  --out examples/necking_bar_quarter_refined_hex8.msh
python examples/generate_necking_prism_gmsh.py --element hex8 \
  --out examples/necking_prism_small_hex8.msh
python examples/generate_necking_prism_gmsh.py --element hex20 \
  --out examples/necking_prism_small_hex20.msh
python examples/generate_necking_prism_gmsh.py --element hex8 \
  --nx 4 --ny 4 --nz 48 --out examples/necking_prism_refined_hex8.msh
```

Run a deck from the repository root:

```bash
python -m fe_solver examples/case_a_hex8.toml
python -m fe_solver examples/case_b_hex8_fbar.toml
python -m fe_solver examples/frame_objectivity_hex8.toml
python -m fe_solver examples/periodic_core_isochoric_hex8_fbar.toml
python -m fe_solver examples/periodic_core_shear_hex8_fbar.toml
python -m fe_solver examples/j2_necking_bar_hex8_fbar.toml
python -m fe_solver examples/j2_necking_prism_small_hex8.toml
python -m fe_solver examples/j2_necking_prism_small_hex8_fbar.toml
```

For a controlled partial run, add `--stop-time 0.5`. Relative mesh and output paths are resolved from the deck directory.
The explicit equivalent command is `python -m fe_solver solve DECK`.

Element evaluation defaults to the serial reference path. To reuse two local worker processes throughout the solve:

```bash
python -m fe_solver examples/j2_necking_prism_small_hex8_fbar.toml --num-processes 2
```

The unit of work is one complete element; the solver does not create a process per element or per quadrature point.
Worker completion may be unordered, while global force and stiffness reduction remains in mesh order for reproducible
floating-point behavior. The startup message and `run_log.json` report requested/effective process counts and available
physical/logical CPUs. The solver never chooses all CPUs automatically, and each worker limits nested BLAS pools to one
thread. For CPU-bound element kernels, physical-core count is the conservative starting point; benchmark before using
additional hardware threads. Add `--debug-timing` when detailed element-phase and sparse-finalization timings are needed.

Compare two completed HDF5 runs, including all numerical result fields and metadata, with:

```bash
python -m verification.compare_run_databases REFERENCE_RUN_H5 CANDIDATE_RUN_H5
```

The comparison normalizes only the output-directory entry in the embedded input deck and ignores the expected
Git-revision provenance difference; both exceptions are listed in its JSON report.

Create one ParaView-readable temporal dataset after a run with:

```bash
python -m fe_solver postprocess examples/results/case_a_hex8/run.h5
```

This writes an `.xdmf` entry point and an `.xdmf.h5` sidecar with connectivity and virtual-dataset views of time slices.
Field values remain in the solver's `run.h5`; the views store selection metadata, not copies of the values. Keep all three
files together (or retain their relative paths). No visualization files are produced during solution. Ordinary HDF references
avoid XML HyperSlab compatibility problems in ParaView's XDMF3 reader.

Stress and both strain measures are stored as six components in the order **`11,22,33,12,23,13`**.
Shear strains are tensor components, **not doubled engineering shears**. HDF5 datasets record this convention in
their attributes. The XDMF exposes only these six-component arrays; select their individual components in ParaView's
component selector. It does not add duplicate scalar aliases such as `cauchy_stress_11`. Six-component array magnitudes
are not tensor Frobenius norms; contractions require double weighting of shear products and compatible stress/strain
measures. Internal FE assembly conventions have not changed.

The results schema is now version 3; old nine-component databases must be regenerated by rerunning the solver.
Regenerate the XDMF too, then reopen it in ParaView rather than reusing an old reader with cached field shapes.

### Uniaxial stretch followed by rotation

`frame_objectivity_hex8.toml` stretches the rod by 10% through `t=0.1`, then rotates it counterclockwise about z
by 90 degrees through `t=1`. Only the two end faces are driven. They contract according to the exact neo-Hookean
uniaxial-stress solution; the four lateral faces are traction-free. The interior and free-side displacements are solved,
not prescribed. The material-specific `neo_hook_uniaxial_rotation` boundary path uses the named material's properties
and remains exact between output events, including cutbacks.

```bash
python -m fe_solver examples/frame_objectivity_hex8.toml
python -m fe_solver postprocess examples/results/frame_objectivity_hex8/run.h5
```

Open `examples/results/frame_objectivity_hex8/run.xdmf`. Use Warp By Vector with `displacement` and scale 1.
At `t=0.1`, component `11` of `cauchy_stress` is positive and component `22` is approximately zero. At `t=1`, those values
exchange places. In the six-component `cauchy_stress` array, these are indices **0 and 1**. Spatial Almansi strain
rotates too, but transverse contraction means its lateral normal components are negative, not zero.
Green-Lagrange strain remains fixed in the material frame during rotation.

### Finite-strain J2 necking benchmark

`j2_necking_bar_hex8_fbar.toml` models one eighth of the classical imperfect circular tensile bar. Its quarter cross-section uses a regular central square and two transfinite outer sectors, avoiding collapsed axis elements and the skewed 45-degree surface cells of a one-block square-to-disk map. Half of the axial layers lie in the central third where necking begins. The prescribed 7 mm motion is applied to the end of the half-model. Material properties are written in the consistent mm--N--MPa system.

After solving, extract end reaction, middle radius, monitor-point radial displacement, and maximum equivalent plastic strain with:

```bash
python -m verification.check_j2_necking examples/results/j2_necking_bar_hex8_fbar/run.h5
python -m verification.check_j2_necking examples/results/j2_necking_bar_hex8_fbar/run.h5 \
  --plot /tmp/j2_necking_bar.png
```

The HDF5 database stores the inverse plastic metric and equivalent plastic strain at element centroids. J2 equivalent stress is intentionally not stored because it can be calculated from the six Cauchy-stress components in ParaView.

The 96-element `j2_necking_prism_small_hex8_fbar.toml` case is a qualitative nonlinear-solver diagnostic, not a replacement for the circular benchmark. `j2_necking_prism_small_hex8.toml` uses the identical mesh, material, loading, and solver controls as an unstabilized standard-Hex8 comparison. All J2 necking decks enable residual-based Newton backtracking. Invalid intermediate configurations reject only the current search length; pseudo-time cutback remains the fallback when no admissible decreasing search step exists.

Extract the square-prism response or compare the two formulations with:

```bash
python -m verification.check_j2_prism \
  examples/results/j2_necking_prism_small_hex8_fbar/run.h5 --summary-only
python -m verification.check_j2_prism \
  examples/results/j2_necking_prism_small_hex8_fbar/run.h5 \
  --compare examples/results/j2_necking_prism_small_hex8/run.h5 \
  --summary-only --output /tmp/j2_prism_comparison.json \
  --plot /tmp/j2_prism_comparison.png
```

The comparison reports reaction, transverse contraction, plastic localization, raw and material-seen Jacobian ranges,
and cross-section mean-stress variation. Its optional six-panel plot places the physical response and volumetric
diagnostics against prescribed end elongation. The standard Hex8 result is a locking control; convergence alone does
not make it a reference solution. A plot from the first completed paired 96-element runs is retained under
`verification/j2_prism_results/` as a qualitative regression artifact.

The follow-on formulation study is expressed as orthogonal variants of that base deck rather than duplicated TOML
files. List the cases with:

```bash
python -m verification.run_j2_formulation_study --list-cases
```

The matrix contains the completed 2x2x24 Hex8/Hex8-Fbar pair, a paired 4x4x48 mesh refinement, a coarse pair with the
bulk modulus halved from 164210 to 82105 MPa, a 2x2x24 serendipity Hex20 case, and the 960/7680-element circular-bar
mesh pair. The half-bulk case changes the elastic Poisson ratio from approximately 0.290 to 0.132 while retaining every
shear/plastic parameter. Run one new case with, for example:

```bash
python -m verification.run_j2_formulation_study \
  --case soft_bulk_hex8_fbar --num-processes 2 --debug-timing --stop-time 0.5
```

New variant outputs are isolated under `examples/results/j2_formulation_study/`; the three baseline cases reuse their
existing result directories. Use `check_j2_prism` to compare any completed pair directly from HDF5; this avoids
ParaView interpolation, warp scaling, camera, and color-range ambiguity. The refined cases are intentionally expensive
and should use the long-run workflow rather than the ordinary test suite.

Standalone material-point characterization and evolved-state tangent diagnostics are available through:

```bash
python -m verification.run_material_point_characterization
python -m verification.check_evolved_tangents DECK RESTART RUN_H5 --trial-time 0.405 --output report.json
python -m verification.diagnose_newton_conditioning DECK RESTART RUN_H5 --committed-time 0.43 --attempt-time 0.44 --line-search none --output report.json
```

The first command writes reviewable J2 uniaxial and simple-shear CSV, compressed NumPy, PNG, and JSON results under `verification/material_point_results/`. The conditioning utility deliberately uses a dense null-space/SVD and is restricted to small diagnostics; production assembly and KKT solution remain sparse.

The isolated J2 JIT feasibility experiment uses the optional `j2-jit`
dependency and leaves the ordinary solver on the interpreted backend.  Run the
synthetic material benchmark with:

```bash
python -m verification.run_j2_numba_feasibility material
```

The runner can also produce separate, directly comparable small-prism results:

```bash
python -m verification.run_j2_numba_feasibility prism \
  --backend python --num-processes 2 --stop-time 0.02
python -m verification.run_j2_numba_feasibility prism \
  --backend numba --num-processes 2 --stop-time 0.02
```

Use its `compare-prisms` subcommand for numerical, nonlinear-path, and internal
wall-time comparison.  JIT outputs are isolated under
`examples/results/j2_numba_feasibility/` and never overwrite formulation-study
databases.  Once the equivalence checks have passed, select the compiled kernel
for an ordinary J2 solve or formulation-study case with `--j2-backend numba`.
The default remains `python`, so installing the optional dependency alone does
not silently change a result path.

## Verify

The test suite uses the standard library runner, so no separate test dependency is required:

```bash
python -m unittest discover -s tests -v
```

It checks shape functions, material and element tangents, homogeneous F-bar equivalence, Gmsh periodic maps, spanning-tree constraints, nonsymmetric whole-KKT solves, event merging, restart compatibility, forced-failure cutback/rollback, and serial-versus-process assembly equivalence.

Run the complete regression and featured-example acceptance suite with:

```bash
python -m verification.run_acceptance
```

The substantially larger necking benchmark is selectable explicitly and is not part of the default quick suite:

```bash
python -m verification.run_acceptance --deck j2_necking_bar_hex8_fbar.toml
```

When ParaView is installed, also test its real readers and Warp By Vector against the HDF5 database:

```bash
python -m verification.check_paraview examples/results/frame_objectivity_hex8/run.h5 --pvpython /path/to/paraview/bin/pvpython
```

This checks both XDMF3 reader variants at every saved time, all field components, and warped node positions.

## Package layout

- `fe_solver/shape.py`, `quadrature.py`: interpolation and integration
- `fe_solver/materials.py`, `material_point.py`: named material API, material models, and prescribed-F verification driver
- `fe_solver/tangents.py`: shared material-to-spatial tangent transformation
- `fe_solver/elements.py`: standard and F-bar element kernels
- `fe_solver/mesh.py`, `constraints.py`: Gmsh import and affine constraints
- `fe_solver/preprocess.py`, `assembly.py`, `execution.py`, `solver.py`: validated setup, deterministic serial/process element execution, sparse assembly, and nonlinear solution
- `fe_solver/io.py`, `config.py`: HDF5 result/restart I/O and TOML/event handling
- `fe_solver/postprocess.py`: temporal XDMF construction from a completed HDF5 run
