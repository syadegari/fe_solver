# Hex8-Fbar patch test

Centroidal F-bar Hex8 on the same distorted 2×2×2 patch geometry as the Hex8
case (`patch_cube_hex8.msh`, identical node coordinates), driven to the same
finite deformation `F̄` (20% stretch with shear, det 1.1501) over four
increments; see [../hex8/README.md](../hex8/README.md) for the field and
material.

One caveat specific to this element, so the result is not over-claimed: under
a homogeneous deformation the centroid and Gauss-point volume ratios coincide,
so `α ≡ 1` and the volumetric projection is inactive. This case certifies that
F-bar *preserves* the constant state — including its residual, which uses
`σ = P F̄ᵀ/J_c` and the F-bar `B̄` tangent machinery — while the projection
itself is verified by the independent spatial-versus-reference kernel
comparison in `tests/test_numerics.py`. The regression suite additionally runs
this patch at κ/μ = 10⁴ (near-incompressible, the element's design point).

```bash
python -m fe_solver solve verification/patch_tests/hex8_fbar/patch_hex8_fbar.toml
```

Measured solved-field error ~`3e-16`; evidence table and visualization:
[../README.md](../README.md).
