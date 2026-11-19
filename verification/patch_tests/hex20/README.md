# Hex20 patch test

20-node serendipity Hex20 on the distorted 2×2×2 patch (`patch_cube_hex20.msh`):
81 nodes, 8 elements with genuinely *curved* interior edges — mid-edge nodes
are perturbed off the straight-edge midpoints — while boundary mid-nodes stay
inside their face planes. Driven to the same finite `F̄` as the Hex8 case
(see [../hex8/README.md](../hex8/README.md)); the isoparametric basis contains
`X` itself, so the affine field is exact even on curved-edge geometry.

This is the strongest case of the three for quadrature: the patch integrand
`∇_X N_A · det J₀` reaches polynomial degree 5 per variable here, and the
production 3×3×3 Gauss rule is exact to degree 5 — the minimum rule that
passes, with no slack (a 2×2×2 rule fails at ~2e-2, but is not selectable:
quadrature is fixed per formulation). Unlike Hex8, this patch has 7 interior
nodes (21 free DOFs): the centre node plus six interior mid-edge nodes.

```bash
python -m fe_solver solve verification/patch_tests/hex20/patch_hex20.toml
```

Measured solved-field error ~`4e-16`, two Newton iterations per increment.
Evidence table and visualization: [../README.md](../README.md).
