# Finite-strain FE prototype

This repository implements the v1 contract in `docs/IMPLEMENTATION_SPEC.md`:

- full-integration Hex8 and Hex20 serendipity elements;
- centroidal F-bar Hex8 with its complete projection tangent;
- compressible neo-Hookean material through the endpoint `F_n`/`F_np1` material-point API;
- sparse COO/CSR assembly and a whole-system sparse-LU KKT solve for affine constraints;
- exact and modified Newton, recoverable trial rollback, adaptive cutback, and mandatory events;
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
```

Run a deck from the repository root:

```bash
python -m fe_solver examples/case_a_hex8.toml
python -m fe_solver examples/case_b_hex8_fbar.toml
python -m fe_solver examples/frame_objectivity_hex8.toml
python -m fe_solver examples/periodic_core_isochoric_hex8_fbar.toml
python -m fe_solver examples/periodic_core_shear_hex8_fbar.toml
```

For a controlled partial run, add `--stop-time 0.5`. Relative mesh and output paths are resolved from the deck directory.
The explicit equivalent command is `python -m fe_solver solve DECK`.

Create one ParaView-readable temporal dataset after a run with:

```bash
python -m fe_solver postprocess examples/results/case_a_hex8/run.h5
```

This writes an `.xdmf` entry point and its small connectivity-ordering sidecar. Field values remain in the solver's `run.h5`; no visualization files are produced during solution.

## Verify

The test suite uses the standard library runner, so no separate test dependency is required:

```bash
python -m unittest discover -s tests -v
```

It checks shape functions, material and element tangents, homogeneous F-bar equivalence, Gmsh periodic maps, spanning-tree constraints, nonsymmetric whole-KKT solves, event merging, restart compatibility, and forced-failure cutback/rollback.

Run the complete regression and featured-example acceptance suite with:

```bash
python -m verification.run_acceptance
```

## Package layout

- `fe_solver/shape.py`, `quadrature.py`: interpolation and integration
- `fe_solver/materials.py`: named material API and neo-Hookean model
- `fe_solver/elements.py`: standard and F-bar element kernels
- `fe_solver/mesh.py`, `constraints.py`: Gmsh import and affine constraints
- `fe_solver/preprocess.py`, `assembly.py`, `solver.py`: validated setup, sparse assembly, and nonlinear solution
- `fe_solver/io.py`, `config.py`: HDF5 result/restart I/O and TOML/event handling
- `fe_solver/postprocess.py`: temporal XDMF construction from a completed HDF5 run
