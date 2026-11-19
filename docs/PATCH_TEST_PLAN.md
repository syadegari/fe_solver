# Patch tests for Hex8, Hex8-Fbar, and Hex20

Work plan for the README item *"Add patch tests for Hex8, Hex20, and Hex8-Fbar"*.
Agreed in discussion on 2026-11-16; the supporting measurements below were run
before implementation and are reproduced by the committed tests.

## 1. What a patch test certifies here — and what it does not

A patch of distorted elements is driven to an exact homogeneous finite
deformation `u = (F̄ − I)X`. Because the isoparametric basis contains `X`
itself, every element type represents this field exactly, so the discrete
solution must reproduce it *to machine precision*: constant `F = F̄` at every
Gauss point, constant stress, and zero residual at every interior node.

The test certifies the **residual chain**: shape functions → kinematics
(`J0`, `Jx`, `grad_X`, `grad_x`, `F`) → stress transform `σ = J⁻¹PFᵀ` → `B` →
quadrature volume → assembly → constraint rows → KKT solve. It says nothing
about the **tangent** — a wrong `K` changes the convergence rate, not the
converged answer. The tangent is covered by the directional-derivative tests in
`tests/test_numerics.py` and the independent F-bar reference kernel.

The gap this closes: every existing exact-solution test runs on a *regular*
mesh (Case B is a structured cube), and regular meshes hide exactly the errors
distortion exposes — Jacobian/inverse-transpose slips, `grad_X` vs `grad_x`
confusion, ordering mismatches. Hex20 has no multi-element exact-solution test
at all before this work.

One line on F-bar, so the result is not over-claimed: under a homogeneous
deformation `J_c = J_g`, so `α ≡ 1` and the volumetric projection is inactive —
the patch test confirms F-bar does not *break* the constant state; the
projection itself is validated by the spatial-versus-reference kernel
comparison in `test_numerics.py`.

## 2. Why machine precision is the right tolerance (quadrature exactness)

The interior-node identity `Σ_e ∫ σ·∇ₓN_A dv = 0` is exact only if the
production quadrature integrates `∇_X N_A · detJ₀` exactly on distorted
geometry. Degree count per parent variable (column `j` of `J0` is `∂X/∂ξ_j`,
so one differentiation drops its degree in `ξ_j`):

| | shape `N` | `J0` col j (in ξ_j / others) | cofactor (in ξ_j / others) | integrand |
|---|---|---|---|---|
| Hex8, 2×2×2 (exact ≤ 3) | 1 | 0 / 1 | 2 / 1 | **2** |
| Hex20, 3×3×3 (exact ≤ 5) | 2 | 1 / 2 | 4 / 3 | **5** |

Hex20 at 3×3×3 is exactly tight — the minimum rule that passes the patch test.
Measured on randomly distorted (including curved-edge Hex20) geometry against a
10-point reference: both rules exact to ~2e-15; one order lower fails at ~2e-2.
The rule is not selectable — `elements._quadrature()` fixes it per formulation
— so reduced integration is unreachable from a deck.

All assertions are therefore set at machine-precision scale. The thresholds are
not a theoretical bound: they are float64 round-off (`eps ≈ 2.2e-16`)
accumulated over the element arithmetic (inverse, determinant, 8–27 Gauss-point
sums), measured at `1e-15`–`1e-13` relative, with two to three decades of
margin. Measured values are recorded in
[../verification/patch_test_results/README.md](../verification/patch_test_results/README.md).

## 3. Patch geometry

One mesh per element type: the canonical minimal patch. Unit cube `[0,1]³`,
**2×2×2 = 8 elements**, built in memory (Section 5).

- Interior nodes: perturbed by up to `0.18·h` in all three directions.
- Boundary nodes: perturbed only tangentially within their face (faces stay
  planar, cube edges straight); cube corners fixed.
- Hex20: mid-edge nodes start at the perturbed straight-edge midpoints and are
  then perturbed by the same rule, so interior edges are genuinely *curved* —
  the case that is exactly tight in the table above. Boundary-face mid-edge
  nodes stay in their face plane.
- Classification (which directions a node may move) uses the node's
  *unperturbed* lattice position.

A larger patch (3×3×3) was considered and rejected: the patch identity is
per-node — it depends on per-element quadrature exactness, field homogeneity,
and the DOF map, none of which scale with element count — so a larger patch
only resamples the same identity. The budget goes instead into **independent
distortion seeds and several `F̄`**, which sample geometry space, where failure
is at least conceivable. Random per-node offsets also carry no reflection or
point symmetry, so errors from "opposite" elements cannot cancel by
construction; running at least two seeds per element type guards the residual
risk of an accidentally benign draw.

Note the minimal patch is only "small" for Hex8 (1 interior node, 3 free
DOFs); the Hex20 patch has 7 interior nodes (centre + 6 interior mid-edge
nodes) = 21 free DOFs.

## 4. Test inventory

All in `tests/test_patch.py`, part of the standard unit suite.

### PT-1a — Consistency patch test (no solve) — the sharp statement

Set `u = (F̄ − I)X` at every node analytically, call `assemble_internal`, and
assert per formulation × seed × `F̄`:

- residual at interior DOFs ≤ 1e-11 × force scale;
- `max|F_gauss − F̄|` ≤ 1e-12 at every Gauss point of every element;
- Cauchy-stress spread across all Gauss points ≤ 1e-11 × `max|σ|`;
- `J = det F̄` everywhere;
- the **full** residual matches the independent analytic value
  `f_A = P : ∫∇_X N_A dV₀` to 1e-11 relative.

Being solver-free, this is not limited by any Newton stopping tolerance.

The analytic oracle was added after the teeth check (Section 4.6) showed the
interior-residual identity alone is blind to any defect that leaves the stress
merely *constant* rather than correct — `Σ_e ∫ σ̃·∇ₓN_A dv = 0` holds for any
constant `σ̃`. The oracle is built from the shape functions and the material
only, never the element kernel's stress transform, `B` matrix, Voigt packing or
volume measure.

Cases:

| case | formulations | material | `F̄` |
|---|---|---|---|
| finite stretch+shear | hex8, hex8_fbar, hex20 | neo-Hookean μ=1, κ=20 | `F_A` below |
| near-incompressible | hex8_fbar | neo-Hookean μ=1, κ=1e4 | `F_A` |
| stateful, yielded | hex8, hex8_fbar | J2 steel (test property set) | `diag(1.05, 0.977, 0.977)` |

```
F_A = [[ 1.20,  0.15, -0.08],
       [ 0.05,  0.92,  0.10],     det = 1.1501
       [-0.04,  0.07,  1.06]]
```

`F_A` is deliberately non-symmetric (a transpose slip cannot pass), genuinely
finite (20% axial), with shear and volume change. The J2 case additionally
asserts the committed trial state is homogeneous and actually yielded.

### PT-2 — Rigid-body patch test

`F̄ = R(z, 90°)` composed with translation `[0.3, −0.2, 0.15]`, same harness.
Stress is zero, so assertions are absolute, scaled by μ: `max|σ| ≤ 1e-12·μ`
and the **entire** residual (not only interior) ≤ 1e-12·μ. Covers all three
formulations; catches objectivity errors in assembly/geometry that the
single-element translation test cannot.

### PT-1b — Dirichlet patch test, solved end-to-end

Drive the same patch through `run_analysis`: one `[[constraints.affine]]`
entry listing all six faces with `macro_F` ramped `I → F_A` over `t ∈ [0,1]`
in 4 increments, and reuse the existing `verification.affine_deformation`
check. Newton tolerances are tightened well past the usual deck values
(`force_rtol = 1e-12`, `force_atol = 1e-14`, constraint tolerances `1e-14`)
so the measured error reports the element's accuracy, not the solver's
stopping rule; `affine_tolerance = 1e-10`. Assert convergence with no cutback
and a bounded iteration count. Backend `scipy_splu` only — deterministic.

### RT — Element rank test

The Dirichlet patch test structurally cannot see a spurious zero-energy mode:
with every boundary node prescribed there are too few free DOFs for one to
develop, at any patch size. Companion check on a single distorted element at
the undeformed state: `K` has **exactly 6** zero modes (eigenvalue magnitude
< 1e-8 × max) and the 7th eigenvalue is well separated (measured ≥ 1e-3 × max
for all three formulations; asserted ≥ 1e-5). F-bar's `K` is non-symmetric on
distorted geometry even at zero stress (`B ≠ B̄`, measured ‖K−Kᵀ‖/‖K‖ ≈ 5e-2),
so eigenvalue magnitudes of the non-symmetric matrix are used.

### Self-guards (inside the tests, not separate tests)

1. **The mesh is actually distorted**: minimum interior-node offset from the
   reference lattice above a floor, and faces still planar / reference
   Jacobians positive. Guards against a future regeneration silently
   degenerating the test into the regular-grid case Case B already covers.
2. **The patch has free interior DOFs** (at least one interior node).
3. **The J2 case actually yielded** (`equivalent_plastic_strain > 0`).

### 4.6 Teeth check — once, recorded, not committed

Each candidate defect must be *reachable* — something a future edit could
plausibly introduce — or it proves nothing. Reduced integration for Hex20 was
considered and dropped on that ground: quadrature is coupled to the formulation
in `elements._quadrature()` with no deck knob, so it cannot be selected.

Five reachable defects were injected by patching module-level helpers for one
assembly (`grad_X` transpose slip, reference instead of current volume measure,
Voigt shear swap, `B`-row swap, a mistyped Gauss weight), each on a distorted
and a regular patch. Full table in the verification note. Two results shaped
the final test:

- Three defects pass the interior-residual criterion and are caught only by the
  analytic oracle — hence its addition above.
- The `grad_X` transpose slip produces **bit-identical** output on a regular
  mesh (`J0` is diagonal there, so `J0⁻¹ = J0⁻ᵀ`). Without distortion the patch
  test is not weaker for this defect class, it is void — which is the concrete
  justification for the distortion self-guard.

No permanent mutation test is committed.

## 5. Mesh and deck construction: in-memory, no new files

`run_analysis` accepts a `PreparedAnalysis`, so the end-to-end test does not
need a `.msh` file: the test builds the `Mesh` dataclass directly (~80 lines,
readable in one sitting), writes its deck as a TOML string loaded through
`load_deck` (full deck validation), and assembles
`PreparedAnalysis(deck, mesh, build_model(...), build_constraints(...),
mandatory_events(...))`. This was chosen over a Gmsh generator + committed
mesh as the smallest-surface-area option. Costs accepted: `read_gmsh` is not
exercised here (it is exercised by every mesh-based test already), and no new
`examples/` decks or acceptance registrations are added — the patch test is
mathematical verification and lives in the unit suite, per spec §22.

## 6. Deferred: force/traction patch test (PT-3)

The strong-form companion — minimal rigid-body constraints plus consistent
boundary loads from the constant-`P` state, recover the field — is deferred.
What is missing to enable it:

1. **A per-node or traction load path.** Spec §12 admits only dead nodal
   forces (one amplitude per region) and body force; consistent nodal loads
   differ node by node (negative corner weights on Hex20). Enabling PT-3 means
   either a surface-traction load type (reference-area, dead — a constant `P̄`
   on the reference boundary is configuration-independent, so this stays
   within the no-follower-load rule) or per-node dead loads.
2. **Non-circular load generation.** Taking `f_int` at the exact solution as
   the load would test the code against itself. The honest version integrates
   `P̄·N` analytically over the planar cube faces, independent of the element
   kernels.

Its unique value over PT-1a/1b: it detects over-constraint and spurious modes
under minimal supports. The rank test (RT) covers the spurious-mode half at
much lower cost, which is why PT-3 is not in this work item.

## 7. Execution

- Run in the `py3.14` conda environment; submit through `rmgr`
  (`rmgr submit ... -- conda run -n py3.14 python -m unittest tests.test_patch -v`).
- Resource needs are trivial: ≤ 2 cores, well inside the AGENTS.md budget;
  wall time seconds.

## 8. Completion checklist

- [x] `tests/test_patch.py`: PT-1a, PT-2, RT, self-guards.
- [x] Analytic residual oracle (added in response to the teeth check).
- [x] PT-1b solved variant through `run_analysis` on a `PreparedAnalysis`.
- [x] Teeth check performed and recorded in
      [../verification/patch_test_results/README.md](../verification/patch_test_results/README.md).
- [x] README: tick the patch-test item; add a Verification-table row.
- [x] Spec §22.1: add the patch test to the required verification list.
- [x] Measured tolerances recorded in the verification note.

Follow-on raised by this work, tracked separately in the README: dead
surface-traction loads (Section 6), whose verification would include a stress
patch test and a published 3D reference example.
