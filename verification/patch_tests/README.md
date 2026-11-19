# Patch tests

Code-verification cases for the element technology, distinct from the software
regression suite: each subdirectory is a frozen, self-contained simulation —
a committed mesh and deck, run exactly like any other deck — whose solution is
known in closed form. A distorted 2×2×2 unit-cube patch is driven on all six
faces to the homogeneous finite deformation `u = (F̄ − I)X`, which every
isoparametric element represents exactly, so the solver must reproduce it to
machine precision. Design, rationale, and what these tests do *not* certify:
[../../docs/PATCH_TEST_PLAN.md](../../docs/PATCH_TEST_PLAN.md). Recorded
margins and the teeth-check study:
[../patch_test_results/README.md](../patch_test_results/README.md).

| case | element | inputs |
| --- | --- | --- |
| [`hex8/`](hex8/) | full-integration Hex8 | `patch_hex8.toml`, `patch_cube_hex8.msh` |
| [`hex8_fbar/`](hex8_fbar/) | centroidal Hex8-Fbar | `patch_hex8_fbar.toml`, `patch_cube_hex8.msh` |
| [`hex20/`](hex20/) | serendipity Hex20 | `patch_hex20.toml`, `patch_cube_hex20.msh` |

## Running a case

From the repository root, exactly as for any simulation:

```bash
python -m fe_solver solve verification/patch_tests/hex8/patch_hex8.toml
```

The deck's `[verification]` table makes the run itself the first check: the
solver verifies at every accepted increment that the whole displacement field
matches the affine solution to `1e-12` and aborts otherwise. Results are
written to `hex8/results/` (ignored by Git).

## Verification evidence

To run all cases and print every measured quantity against its threshold —
interior residual, Gauss-point `F` and `J` errors, Cauchy-stress spread, the
independent analytic-oracle residual error, solved-field error, Newton
iteration counts:

```bash
python -m verification.patch_tests.verify            # or: verify hex8 hex20
```

A nonzero exit status means a threshold was exceeded.

## Visualization

For ParaView, postprocess a completed run and open the XDMF:

```bash
python -m fe_solver postprocess verification/patch_tests/hex8/results/run.h5
```

With `run.xdmf` loaded: no filter shows the reference configuration (geometry
is reference coordinates; displacement is a point field); **Warp By Vector**
at scale 1 shows the deformed patch — the imposed deformation is finite, so no
exaggeration is needed; adding the **Shrink** filter (factor ≈ 0.8) after the
warp gives the exploded view that separates the elements and exposes the
off-lattice interior node and the element distortion.

## Relation to the regression suite and CI

`tests/test_patch.py` is the always-on counterpart, run with the normal unit
suite and in CI. It shares this directory's construction code
(`patch_cube.py`) and asserts the same thresholds, plus the cases that need no
frozen deck: assembled-residual consistency at the exact field (free of any
Newton tolerance), rigid-rotation objectivity, a yielded-J2 state, a
near-incompressible F-bar variant, element rank (exactly six zero-energy
modes), and mesh-distortion self-guards. Its `FrozenInputTests` pins the files
here to the generator, so the frozen inputs cannot drift silently.

## Housekeeping

`generate_patch_cases.py` regenerates the meshes and decks and round-trips
them through the production Gmsh reader; it exists so changes to the frozen
inputs stay traceable, and is not part of running the tests:

```bash
python -m verification.patch_tests.generate_patch_cases
```
