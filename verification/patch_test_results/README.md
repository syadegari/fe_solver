# Patch-test results and teeth check (2026-11-17)

Record for `tests/test_patch.py`; the design is in
[../../docs/PATCH_TEST_PLAN.md](../../docs/PATCH_TEST_PLAN.md).

A distorted 2x2x2 unit-cube patch is driven to an exact homogeneous finite
deformation `u = (Fbar - I) X`, which every isoparametric element represents
exactly, so the discrete answer must reproduce it to machine precision.

## 1. Achieved margins

Relative to the local force / stress scale; assertion thresholds are `1e-11`
(residual, stress spread, analytic oracle) and `1e-12` (`F`, `J`).

| case | seed | interior residual | `F` error | `J` error | stress spread | analytic oracle |
|---|---|---|---|---|---|---|
| hex8 | 11 / 29 | 1.2e-15 / 8.8e-16 | 9.9e-16 / 6.7e-16 | 8.9e-16 | 8.8e-15 / 8.1e-15 | 1.6e-15 / 1.5e-15 |
| hex8_fbar | 11 / 29 | 1.5e-15 / 8.8e-16 | 9.9e-16 / 6.7e-16 | 8.9e-16 | 8.6e-15 / 7.4e-15 | 1.7e-15 / 1.6e-15 |
| hex8_fbar, κ/μ=1e4 | 11 / 29 | 1.6e-15 / 8.8e-16 | 9.9e-16 / 6.7e-16 | 8.9e-16 | 9.4e-15 / 8.0e-15 | 1.6e-15 / 1.5e-15 |
| hex20 | 11 / 29 | 5.2e-15 / 4.4e-15 | 5.3e-15 | 5.1e-15 | 4.2e-14 | 5.7e-15 / 4.6e-15 |
| hex8, J2 yielded | 11 / 29 | 3.6e-14 / 6.1e-14 | 6.2e-16 | 8.9e-16 | 3.3e-13 / 2.4e-13 | 5.3e-14 / 6.1e-14 |
| hex8_fbar, J2 yielded | 11 / 29 | 9.2e-14 / 1.2e-13 | 6.2e-16 | 8.9e-16 | 2.4e-13 / 2.9e-13 | 9.2e-14 / 1.2e-13 |

Tolerances are not derived from a theoretical bound: they are float64 round-off
(`eps ≈ 2.2e-16`) accumulated over the element arithmetic — the inverse,
determinant and 8–27 Gauss-point sums — measured at `1e-15`–`1e-13` relative,
with two to three decades of margin applied.

Solved patch (PT-1b, full deck → KKT → Newton path, four increments):

| formulation | `max |u − u_exact|` | deck `affine_deformation` check | Newton iterations | cutbacks |
|---|---|---|---|---|
| hex8 | 5.6e-17 | 5.6e-17 | 2 | 0 |
| hex8_fbar | 3.2e-16 | 3.2e-16 | 2 | 0 |
| hex20 | 2.0e-16 | 2.0e-16 | 2 | 0 |

## 2. Quadrature exactness

The interior-node identity `Σ_e ∫ σ·∇ₓN_A dv = 0` is exact only if the
production rule integrates `∇_X N_A · detJ₀` exactly on distorted geometry.
Measured polynomial degree of that integrand, by fitting: **2 per variable for
Hex8, 5 for Hex20** (one differentiation drops the degree of `J0` column *j* in
`ξ_j`, which a naive count misses). Against a 10-point reference on a
curved-edge distorted Hex20:

| rule | relative error | |
|---|---|---|
| Hex8, 1×1×1 | 3.5e-2 | inexact |
| Hex8, 2×2×2 (production) | 1.8e-15 | exact, one degree of slack |
| Hex20, 2×2×2 | 1.6e-2 | inexact |
| Hex20, 3×3×3 (production) | 1.6e-15 | exact, **no slack** |

Hex20 at 3×3×3 is the minimum rule that passes the patch test. Note the
quadrature rule is not selectable: `elements._quadrature()` fixes it per
formulation and `quadrature.py` defines the orders as module constants, so a
reduced-integration Hex20 is unreachable from a deck. If selectable reduced
integration is ever added, this patch test will reject it for Hex20.

## 3. Teeth check

Reachable defects were injected by patching module-level helpers
(`_kinematics`, `_B_matrix`, `_stress_voigt`, the weight tables) for the
duration of one assembly; the repository was not modified. Each was run on
both the distorted patch and a regular one. "interior only" is the
interior-residual/`F`/stress-spread criterion; "with oracle" adds the
independent analytic residual.

| injected defect | distorted, interior only | distorted, with oracle | regular mesh |
|---|---|---|---|
| `grad_X` transpose slip | **caught** (2.6e-1 … 7.5e-1) | caught | **missed** |
| `dv` from reference instead of current volume | missed | **caught** (1.3e-1) | caught |
| Voigt shear-component swap in the stress | missed | **caught** (8.2e-3) | caught |
| `B`-matrix shear-row swap | missed | **caught** (8.2e-3) | caught |
| one mistyped Gauss weight | caught (8.6e-4) | caught | hex20 only |

Two findings drove changes to the test:

**The interior-residual identity is blind to any defect that leaves the stress
merely constant rather than correct.** `Σ_e ∫ σ̃·∇ₓN_A dv = 0` holds for *any*
constant `σ̃`, so a wrong-but-uniform stress, a Voigt mis-pairing, or a uniform
rescaling of the volume measure all pass. The `dv` case is the clearest: under
a homogeneous deformation `detJx = J·detJ₀` with `J` constant, so substituting
the reference volume scales the whole residual by a constant and the identity
still holds. The test therefore also compares the **full** residual against the
independent analytic value `f_A = P : ∫∇_X N_A dV₀`, built from the shape
functions and the material only — never the element kernel's stress transform,
`B` matrix, Voigt packing or volume measure. That closes all three blind spots.

**A regular mesh cancels the `grad_X` transpose slip exactly.** On the
undistorted lattice `J0` is diagonal, so `J0⁻¹ = J0⁻ᵀ` and the defect is not
merely hidden — it produces bit-identical output, and no assertion of any kind
can detect it. This is the concrete justification for the distortion guard in
`PatchMeshGuardTests`: without distortion the patch test is not weaker, it is
void for this class of defect.

Candidates considered and dropped: reduced integration for Hex20 is not
reachable (Section 2).

## 4. What the patch test does and does not certify

Certifies the residual chain — shape functions, kinematics, stress transform,
`B`, Voigt packing, volume measure, quadrature, assembly, constraint rows, KKT
solve — on geometry where a regular mesh would hide errors.

Does not certify the tangent: a wrong `K` changes the convergence rate, not the
converged answer. That is covered by the directional-derivative tests in
`tests/test_numerics.py` and the independent F-bar reference kernel.

Does not certify F-bar's volumetric projection: a homogeneous deformation gives
`J_c = J_g`, so `α ≡ 1` and the projection is inactive. The patch test confirms
F-bar does not break the constant state; `test_fbar_spatial_and_reference_forms_agree`
covers the projection itself.

Spurious zero-energy modes are invisible to a Dirichlet patch test at any patch
size, because prescribing every boundary node leaves too few free DOFs for one
to develop. `ElementRankTests` covers that separately: all three formulations
have exactly six zero modes, with the seventh eigenvalue at 4.7e-3 (hex8,
hex8_fbar) and 1.2e-3 (hex20) of the largest. F-bar's tangent is 4.8e-2
non-symmetric on distorted geometry even at zero stress (`B ≠ B̄`), consistent
with the no-symmetry invariant.

## 5. Reproducing

The frozen simulation cases live in
[../patch_tests/](../patch_tests/README.md); run one as an ordinary
simulation, or regenerate every number in Section 1 with the evidence runner:

```bash
python -m fe_solver solve verification/patch_tests/hex8/patch_hex8.toml
python -m verification.patch_tests.verify
```

The regression counterpart is `python -m unittest tests.test_patch -v`. The
teeth check is a one-off experiment, not a committed mutation test; its script
is reproduced from the table above by patching the named helpers.
