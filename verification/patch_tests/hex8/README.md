# Hex8 patch test

Full-integration trilinear Hex8 on a distorted 2×2×2 unit-cube patch
(`patch_cube_hex8.msh`): 27 nodes, 8 irregular elements, one interior node
pushed off the lattice; boundary nodes are perturbed only within their faces,
so the cube faces stay planar. The deck drives all six faces to

```
F̄ = [[ 1.20,  0.15, -0.08],
     [ 0.05,  0.92,  0.10],      det F̄ = 1.1501
     [-0.04,  0.07,  1.06]]
```

ramped over `t ∈ [0, 1]` in four increments, with a compressible neo-Hookean
material (μ = 1, κ = 20) and Newton tolerances tightened to `1e-12`/`1e-14`.

The exact solution is the affine field `u = (F̄ − I)X`; the deck's
`[verification]` table aborts the run if the solved field deviates from it by
more than `1e-12` at any accepted increment. Measured error is ~`1e-16`, with
constant `F`, constant Cauchy stress, and two Newton iterations per increment;
see [../../patch_test_results/README.md](../../patch_test_results/README.md).

```bash
python -m fe_solver solve verification/patch_tests/hex8/patch_hex8.toml
```

The 2×2×2 Gauss rule integrates the patch integrand (degree 2 per variable)
exactly, with one degree of slack. How to run the evidence table and the
exploded visualization: [../README.md](../README.md).
