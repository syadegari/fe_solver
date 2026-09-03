# Finite-Strain FE Prototype: Implementation Specification v1

## 1. Authority and purpose

This document is the **normative coding contract** for the Python prototype. It intentionally omits derivations.

Mathematical authority: `FORMULATION.tex` / `FORMULATION.pdf` in the handoff package.

If this implementation specification and the formulation reference appear to disagree, do not silently choose one. Isolate the conflict, report it, and resolve it before building dependent code.

The prototype must implement:

- standard full-integration Hex8;
- centroidal F-bar Hex8;
- standard full-integration Hex20;
- finite-strain quasi-static equilibrium;
- a generic finite-deformation material-point interface based on endpoint deformation gradients;
- exact or modified Newton iteration;
- general affine linear displacement constraints using Lagrange multipliers and one sparse KKT system;
- Gmsh-based mesh/region/periodicity preprocessing;
- adaptive pseudo-time increments with mandatory load/output/restart event times;
- committed/trial constitutive state, rollback, cutback, output, and restart.


## 2. Explicit exclusions

Do not implement in v1 unless required to correct a defect in the specification:

- contact;
- follower pressure/traction linearization;
- dynamics or mass matrices;
- thermal fields;
- remeshing/adaptivity;
- mixed `u-p` or `u-p-J` elements;
- reduced/selective integration or hourglass control;
- F-bar stabilization for Hex20;
- distributed-memory assembly;
- non-Gmsh mesh readers;
- crystal plasticity.

An iterative KKT backend is optional. Sparse direct KKT solution is the mandatory correctness path.

---

## 3. Repository/module boundary

The code should keep the following concerns separate. Exact filenames may differ, but the dependency directions should remain narrow.

```text
mesh/              Gmsh import, physical groups, periodic maps, local ordering
quadrature/        Gauss rules
shape/             Hex8 and Hex20 interpolation and derivatives
materials/         material registry, state layouts, reference neo-Hookean model
elements/          standard Hex8/Hex20 and Hex8-Fbar kernels
constraints/       C u = d generation, periodic equivalence classes
assembly/          DOF numbering and sparse global assembly
solver/            Newton, KKT linear solve, increment controller
io/                TOML, results, restart
verification/      tangent and acceptance checks
```

No element kernel should import Gmsh. No material model should know which element formulation called it.

---

## 4. Core data ownership

### 4.1 Nodal geometry and displacement

Store reference coordinates once:

```text
X: float64 [n_node, 3]
```

The global displacement vector contains three DOFs per node:

```text
u_n:        float64 [3*n_node]   # committed
u_trial:    float64 [3*n_node]   # current Newton trial
```

Current coordinates are derived:

\[
\boldsymbol x_{A,n}=\boldsymbol X_A+\boldsymbol u_{A,n},\qquad
\boldsymbol x_{A,n+1}^{(i)}=\boldsymbol X_A+\boldsymbol u_{A,n+1}^{(i)}.
\]

### 4.2 Deformation gradients are not history variables

**Invariant:** `F_n`, `F_np1`, raw or projected, are reconstructed from nodal kinematics whenever an element is evaluated. They are not independently advanced Gauss-point state and are not required in a restart file.

### 4.3 Constitutive state

Each material definition declares a fixed `n_state` before allocation. For each homogeneous element/material block, keep separate committed and trial arrays, for example

```text
state_n:     float64 [n_elem_block, n_gauss, n_state]
state_trial: float64 [n_elem_block, n_gauss, n_state]
```

`n_state = 0` is valid for hyperelasticity.

Only accepted global states are committed. A new Newton iteration must never use the previous Newton iteration's trial state as its starting state.

### 4.4 Named request/response records

Do not use positional tuples for material or element APIs.

Suggested Python records:

```python
@dataclass(frozen=True)
class MaterialRequest:
    F_n: np.ndarray          # (3, 3)
    F_np1: np.ndarray        # (3, 3), current trial endpoint
    state_n: np.ndarray      # (n_state,)
    parameters: object
    point_properties: object | None
    t_n: float
    t_np1: float
    need_tangent: bool

@dataclass
class MaterialResponse:
    P: np.ndarray                    # (3, 3)
    A_alg: np.ndarray | None         # (3, 3, 3, 3), dP_np1/dF_np1
    state_trial: np.ndarray          # (n_state,)
    status: MaterialStatus
```

The order shown here is not a tuple order; access fields by name.

`need_tangent=False` may skip construction of `A_alg`, but **must not alter** `P`, `state_trial`, or `status`.

---

## 5. Material registry and material assignment

### 5.1 Separate model, material definition, and point properties

Do not encode constitutive meaning in a single integer material ID.

Use structured material definitions, conceptually:

```python
@dataclass(frozen=True)
class MaterialDefinition:
    name: str
    model: str
    parameters: Mapping[str, Any]
    state_layout: StateLayout
```

An element assignment maps a named Gmsh volume Physical Group to:

- element formulation (`hex8`, `hex8_fbar`, `hex20`);
- material definition name;
- optional property source.

For crystal plasticity later, one phase/material definition may be shared by many elements while initial orientation is supplied as an element or integration-point property. Do not create one constitutive model definition merely because the orientation differs.

### 5.2 Solver-native material routine names

Use names distinct from Abaqus terminology. Recommended public interface:

```python
@dataclass(frozen=True)
class MaterialInitRequest:
    parameters: object
    point_properties: object | None
    X: np.ndarray              # reference position of material point, shape (3,)
    t0: float

@dataclass
class MaterialInitResponse:
    state0: np.ndarray         # shape (n_state,)
    status: MaterialStatus

initialize_material_state(MaterialInitRequest) -> MaterialInitResponse
evaluate_material_point(MaterialRequest) -> MaterialResponse
```

The default cold start is the stress-free reference configuration with `u_0 = 0`, `F_0 = I`, and material state returned by `initialize_material_state`. The initializer may use point properties such as crystal orientation. Initial stress is zero for the required reference neo-Hookean model. A future material that supports nonzero initial stress would require an explicit extension of this contract rather than an implicit solver-side assumption.

The solver owns stress-measure and tangent transformations. A material returns first Piola-Kirchhoff stress and `dP/dF` only.

---

## 6. Reference material: compressible neo-Hookean

This model is required before any stateful plasticity model.

Use

\[
W(\boldsymbol F)
=\frac{\mu}{2}\bigl(\operatorname{tr}\boldsymbol C-3\bigr)
-\mu\ln J
+\frac{\kappa}{2}(\ln J)^2,
\qquad
\boldsymbol C=\boldsymbol F^\top\boldsymbol F,
\qquad J=\det\boldsymbol F>0.
\]

Return

\[
\boxed{
\boldsymbol P
=\mu\left(\boldsymbol F-\boldsymbol F^{-\top}\right)
+\kappa\ln J\,\boldsymbol F^{-\top}
}
\]

and

\[
\boxed{
A^{\mathrm{alg}}_{iI jJ}
=\mu\,\delta_{ij}\delta_{IJ}
+\kappa(F^{-\top})_{iI}(F^{-\top})_{jJ}
+(\mu-\kappa\ln J)(F^{-\top})_{iJ}(F^{-\top})_{jI}.
}
\]

There is no material state: `n_state = 0`.

If `J <= 0` or the deformation is numerically invalid, return a recoverable trial failure. Do not return NaNs and continue assembly.

Unit tests must compare the analytic `A_alg` with centered finite differences of the complete `P(F)` evaluation.

---

## 7. Gmsh mesh contract

### 7.1 Authority

The Python Gmsh layer is authoritative for:

- node coordinates;
- hexahedral connectivity;
- named Physical Groups;
- periodic slave/master node correspondence.

The mesh layer should remain Python-based even if the mechanical solver is later ported to C++.

### 7.2 File/API expectations

Use Gmsh MSH 4.1 and the official Python API in the prototype.

Supported solid topologies:

- linear 8-node hexahedron (`hex8`, Gmsh element family/type corresponding to Hex8);
- incomplete quadratic 20-node serendipity hexahedron (`hex20`).

Do not assume Gmsh node tags are dense or zero-based. Build dense internal indices and retain mappings to/from original Gmsh tags.

### 7.3 Local node ordering

The element kernels define their own fixed parent-node ordering. The importer must query Gmsh element properties/local reference coordinates and either verify identical ordering or construct an explicit permutation. Do not assume Hex20 edge-node ordering without checking it.

### 7.4 Physical Groups

Named 3D Physical Groups identify material/element assignment regions. Named 2D groups identify boundary regions. Named 0D groups may identify anchors.

Do not invent a second node-set/element-set language inside the solver.

### 7.5 Periodic maps

For v1, accept only translational periodic maps between opposite RVE boundaries. Read the Gmsh periodic node correspondence instead of matching nodes with coordinate tolerances in the FE solver.

The mechanics relation for a retained pair is

\[
\boldsymbol u^+ - \boldsymbol u^-
=
(\overline{\boldsymbol F}-\boldsymbol I)
(\boldsymbol X^+-\boldsymbol X^-).
\]

A general rotational/affine Gmsh periodic transformation is outside v1 and must be rejected with a clear diagnostic.

---

## 8. Shape functions and quadrature

### 8.1 Standard Hex8

- 8 nodes, tri-linear interpolation.
- `2 x 2 x 2` full Gauss integration.
- 8 material points.

### 8.2 Hex8-Fbar

- same Hex8 interpolation;
- same 8 material Gauss points;
- same `2 x 2 x 2` quadrature;
- one element centroid used **only** for volumetric projection;
- no material state, stress, or constitutive update at the centroid.

### 8.3 Hex20

- 20-node quadratic serendipity interpolation;
- `3 x 3 x 3` full Gauss integration;
- 27 material points.

The exact shape functions are in the formulation reference appendix. Unit tests must verify partition of unity, derivative sums, nodal interpolation, and reproduction properties.

---

## 9. Common element kinematics

For an element and parent point `xi`, construct

\[
\mathbf J_0=\frac{\partial\boldsymbol X}{\partial\boldsymbol\xi},
\qquad
\mathbf J_{x,n}=\frac{\partial\boldsymbol x_n}{\partial\boldsymbol\xi},
\qquad
\mathbf J_{x,n+1}^{(i)}=\frac{\partial\boldsymbol x_{n+1}^{(i)}}{\partial\boldsymbol\xi}.
\]

With column gradients,

\[
\nabla_X N_A=\mathbf J_0^{-\top}\nabla_\xi N_A,
\qquad
\nabla_x N_A=(\mathbf J_{x,n+1}^{(i)})^{-\top}\nabla_\xi N_A.
\]

Reconstruct endpoint deformation gradients:

\[
\boxed{
\boldsymbol F_n=\mathbf J_{x,n}\mathbf J_0^{-1},
\qquad
\boldsymbol F_{n+1}^{(i)}=\mathbf J_{x,n+1}^{(i)}\mathbf J_0^{-1}.
}
\]

All spatial gradients, current quadrature volumes, Cauchy stresses, and spatial tangents used in the current residual/Jacobian belong to the **current Newton trial configuration** `Omega_{n+1}^{(i)}`.

---

## 10. Standard Hex8/Hex20 element kernel

### 10.1 Input

```text
X_e             [n_e, 3]
u_e_n           [3*n_e]
u_e_trial       [3*n_e]
state_e_n       [n_gauss, n_state]
material definition / point properties
t_n, t_np1
need_tangent
```

where `(n_e, n_gauss)` is `(8,8)` or `(20,27)`.

### 10.2 Output

Suggested named response:

```python
@dataclass
class ElementResponse:
    f_int: np.ndarray             # (3*n_e,)
    K: np.ndarray | None          # (3*n_e, 3*n_e)
    state_trial: np.ndarray       # (n_gauss, n_state)
    status: ElementStatus
    gauss_output: object | None
```

### 10.3 Algorithm

For every Gauss point:

1. Evaluate `N`, `dN_dxi`, `J0`, committed/trial `Jx`, `grad_X N`, trial `grad_x N`, `F_n`, `F_np1`, and `J = det(F_np1)`.
2. Validate reference and trial geometry according to Section 18.
3. Call `evaluate_material_point` with raw `F_n` and raw `F_np1`.
4. Convert the returned `P` to Cauchy stress:
   \[
   \boldsymbol\sigma=J^{-1}\boldsymbol P\boldsymbol F^\top.
   \]
5. If a tangent is requested, transform `A_alg = dP/dF` to the Truesdell spatial tangent using formulation labels `eq:a-pushforward` and `eq:cT-from-A`:
   \[
   a^{\mathrm{pf}}_{ij km}
   =\frac1J A^{\mathrm{alg}}_{iI kK}F_{jI}F_{mK},
   \]
   \[
   c^{\mathrm{T,alg}}_{ij km}
   =\frac12\left[
   a^{\mathrm{pf}}_{ij km}+a^{\mathrm{pf}}_{ij mk}
   -\delta_{ik}\sigma_{mj}-\delta_{im}\sigma_{kj}
   \right].
   \]
6. Build the ordinary 6-by-`3*n_e` engineering-shear `B` matrix from current gradients. Use exactly the Voigt convention in formulation label `eq:voigt-explicit`.
7. Current quadrature volume:
   \[
   \omega_g^x=\det(\mathbf J_{x,n+1}^{(i)})W_g.
   \]
8. Accumulate
   \[
   \mathbf f^{e,\mathrm{int}} \mathrel{+}=
   \mathbf B_g^\top\boldsymbol\sigma_g^{\mathrm{V}}\omega_g^x,
   \]
   \[
   \mathbf K^{e,\mathrm{mat}} \mathrel{+}=
   \mathbf B_g^\top\mathbf D_g^{\mathrm{T,alg}}\mathbf B_g\omega_g^x,
   \]
   and for node pair `(A,B)`
   \[
   \mathbf K^{e,\mathrm{geo}}_{AB}\mathrel{+}=
   \left[(\nabla_xN_A)^\top\boldsymbol\sigma_g\nabla_xN_B\right]\mathbf I_3\omega_g^x.
   \]
9. Store the returned state only in the element trial buffer.

Return

\[
\boxed{
\mathbf K^e=\mathbf K^{e,\mathrm{mat}}+\mathbf K^{e,\mathrm{geo}}.
}
\]

Do not assume symmetry and do not store only one triangle.

---

## 11. Centroidal Hex8-Fbar kernel

The material API must remain identical to Section 10.

### 11.1 Endpoint projection

At both the committed endpoint and current trial endpoint, reconstruct the raw Gauss-point and centroid deformation gradients.

For each Gauss point

\[
\alpha_g=\left(\frac{J_c}{J_g}\right)^{1/3},
\qquad
\overline{\boldsymbol F}_g=\alpha_g\boldsymbol F_g.
\]

Do this separately for `n` and `n+1,i`.

Call the material routine with

```text
F_n   = Fbar_n_g
F_np1 = Fbar_trial_g
```

The material returns the same `P`, `A_alg`, state, and status fields as any other element. For this element denote them mathematically by `Pbar` and `Abar` only to make the element derivation readable; the software response type is unchanged.

### 11.2 Stress and residual

At a Gauss point

\[
\overline{\boldsymbol\sigma}_g
=J_c^{-1}\overline{\boldsymbol P}_g\overline{\boldsymbol F}_g^\top.
\]

Use **raw current geometry** to build `B`, `grad_x N`, and the current quadrature weight `omega_x[g] = omega_g^x`. Accumulate

\[
\mathbf f^{e,\mathrm{int}}
\mathrel{+}=
\mathbf B_g^\top\overline{\boldsymbol\sigma}_g^{\mathrm V}\omega_g^x.
\]

### 11.3 Consistent spatial tangent

Push `Abar` forward with `Fbar` and `J_c` to obtain the material Truesdell tangent `Dbar_T_alg` using the same transformation as Section 10.

For each node `B`, define

\[
\boldsymbol q_{B,g}
=\frac13\left(\nabla_x N_B|_c-\nabla_x N_B|_g\right).
\]

For component `k = 0,1,2`, define the projected column of the node block:

\[
\overline{\mathbf B}^{\,g}_B[:,k]
=
\mathbf B^{\,g}_B[:,k]
+q_{B,g,k}
\begin{bmatrix}1&1&1&0&0&0\end{bmatrix}^{\!\top}.
\]

This notation is deliberately explicit: `B_B^g` is a 6-by-3 node block; `[:,k]` is programming-array notation used only in this implementation document.

Accumulate

\[
\mathbf K^{e,\mathrm{mat}}_{\bar F}
\mathrel{+}=
\mathbf B_g^\top
\overline{\mathbf D}^{\mathrm{T,alg}}_g
\overline{\mathbf B}_g\,\omega_g^x,
\]

ordinary geometric blocks using `sigma_bar`, and

\[
\boxed{
\mathbf K^{e,\mathrm{proj}}_{AB}
\mathrel{+}=
-\left(\overline{\boldsymbol\sigma}_g\nabla_x N_A|_g\right)
\otimes\boldsymbol q_{B,g}\,\omega_g^x.
}
\]

Return

\[
\boxed{
\mathbf K^e_{\bar F}
=
\mathbf K^{e,\mathrm{mat}}_{\bar F}
+\mathbf K^{e,\mathrm{geo}}_{\bar F}
+\mathbf K^{e,\mathrm{proj}}_{\bar F}.
}
\]

A separate test implementation may use the reference-form effective-stress tangent (`eq:fbar-dPeff`, `eq:fbar-K-reference`) and must agree with the spatial form to numerical precision.

### 11.4 Output naming

For standard elements:

```text
F_raw == F_material
J_raw == J_material
P_material == P_effective
```

For F-bar Hex8 distinguish:

```text
F_raw, J_raw              raw geometric kinematics
F_material, J_material    Fbar and J_c seen by material
P_material                returned Pbar, conjugate to Fbar
P_effective               alpha^-2 Pbar, conjugate to raw F in reference residual
cauchy_stress             sigma_bar used in current residual
```

Do not write a field named only `F` for F-bar output without clarifying which quantity it is.

---

## 12. Dead external loads

The formulation excludes follower-load tangents. v1 may contain two external-load types, both independent of the current unknown configuration.

### 12.1 Dead nodal forces

A nodal force is specified on a named Gmsh point/boundary group and is applied **per selected node**, not interpreted as a total force to be automatically distributed. Its scalar amplitude may use the same constant-plus-curve value expression as prescribed displacement.

Example:

```toml
[[loads.nodal]]
name = "pull_nodes"
region = "load_nodes"
component = "x"
value = { curve = "force_ramp", scale = 1.0 }
```

The load contributes directly to `f_ext` and has zero tangent.

### 12.2 Dead body force per reference volume

An optional body force `b0` is defined per reference volume and assembled with reference quadrature:

\[
\mathbf f^{e,\mathrm{body}}_A
=\int_{\Omega_{0e}}N_A\boldsymbol b_0\,\mathrm dV
\approx\sum_g N_A(\boldsymbol\xi_g)\boldsymbol b_0(\boldsymbol X_g,t)
\det\mathbf J_{0,g}W_g.
\]

It is added to `f_ext` and has zero tangent. The required acceptance cases use `b0 = 0`.

Example deck form:

```toml
[[loads.body]]
region = "solid"
vector = [
  { constant = 0.0 },
  { constant = 0.0 },
  { curve = "gravity_ramp", scale = -9.81 }
]
```

---

## 13. Sparse global assembly

Use three displacement DOFs per node.

Recommended prototype assembly path:

1. gather element global DOF indices;
2. add element internal force directly to the dense global residual vector;
3. accumulate element tangent entries as sparse COO triplets;
4. sum duplicates and convert to CSR after assembly.

Do not densify the global stiffness except in deliberately tiny unit tests.

---

## 14. General affine displacement constraints

Represent all displacement constraints as

\[
\boxed{\mathbf C\mathbf u=\mathbf d(t)}
\]

with sparse `C`.

The same mechanism covers:

- one prescribed displacement DOF;
- arbitrary linear multipoint equations;
- periodic RVE pair equations.

Do not eliminate dependent displacement DOFs from `K`.

### 14.1 Periodic-equivalence reduction

Opposite Gmsh periodic surfaces cause face, edge, and corner pair relations that can contain cycles. Build a graph of periodic-equivalent nodes and retain a spanning tree for each equivalence class before creating constraint rows. This prevents redundant equations.

Periodic constraints retain an arbitrary rigid translation. Add one independent translational anchor, e.g. all three displacement components of the origin/reference node for the supplied cube.

Validate that `C` has independent rows for the acceptance cases.

---

## 15. KKT system

At Newton iteration `i`, form

\[
\mathbf r_u
=\mathbf f^{\mathrm{int}}-\mathbf f^{\mathrm{ext}}
+\mathbf C^\top\boldsymbol\lambda,
\qquad
\mathbf r_c=\mathbf C\mathbf u-\mathbf d.
\]

Solve the complete augmented system

\[
\boxed{
\begin{bmatrix}
\mathbf K & \mathbf C^\top\\
\mathbf C & \mathbf 0
\end{bmatrix}
\begin{bmatrix}
\Delta\mathbf u\\
\Delta\boldsymbol\lambda
\end{bmatrix}
=-
\begin{bmatrix}
\mathbf r_u\\
\mathbf r_c
\end{bmatrix}.
}
\]

The KKT matrix is expected to be indefinite. Do not treat indefiniteness as singularity.

For full-row-rank `C`, local constrained solvability requires the reduced tangent `Z.T @ K @ Z` to be nonsingular, where columns of `Z` span `null(C)`. This is a diagnostic statement; v1 does not need to construct `Z` in the production solve.

Physical constraint reactions are

\[
\boxed{\mathbf f_c=-\mathbf C^\top\boldsymbol\lambda.}
\]

Individual multiplier values are basis/row-scaling dependent. Do not interpret them as unique physical reactions in isolation.

### 15.1 Mandatory linear backend

The required correctness backend is sparse direct LU of the **whole KKT matrix**. Recommended prototype path:

```text
K: CSR
C: CSR
KKT: sparse block -> CSC
factor/solve: scipy.sparse.linalg.splu (or equivalent spsolve path)
```

Do not separately factor `K` and then eliminate constraints.

### 15.2 Optional iterative backend

An iterative KKT backend may be added behind the same interface. Because the generic tangent may be nonsymmetric, do not make MINRES the generic path. GMRES plus an appropriate block/Schur preconditioner, or PETSc/petsc4py field-split machinery, is a later extension.

---

## 16. Newton and material-state logic

### 16.1 Exact Newton

At every global iteration:

- recompute element stresses/trial state;
- recompute element tangents;
- reassemble `K`;
- rebuild/refactor the KKT matrix.

### 16.2 Modified Newton

At every global iteration:

- **still recompute** element stresses, internal force, and trial state from the committed state `n` to the current trial endpoint;
- reuse the selected frozen `K` and KKT factorization according to the policy;
- set `need_tangent=False` on material calls when no tangent refresh is needed.

A new increment attempt must have a valid tangent/factorization before its first correction unless an explicit safe reuse policy is implemented.

### 16.3 State commit

At iteration `i`, always integrate

\[
(\boldsymbol F_n,\mathcal H_n)
\rightarrow
(\boldsymbol F_{n+1}^{(i)},\widehat{\mathcal H}_{n+1}^{(i)}).
\]

Never integrate from trial iteration `i` to trial iteration `i+1`.

On global convergence, commit all Gauss-point trial states atomically. On failure, discard all trial state.

---

## 17. Convergence controls

Use explicit absolute-plus-relative tests. A reference form is

\[
\|\mathbf r_u\|_\infty
\le
\epsilon_f^{\mathrm{abs}}
+
\epsilon_f^{\mathrm{rel}}
\max\left(
\|\mathbf f^{\mathrm{int}}\|_\infty,
\|\mathbf f^{\mathrm{ext}}\|_\infty,
\|\mathbf C^\top\boldsymbol\lambda\|_\infty
\right),
\]

and

\[
\|\mathbf r_c\|_\infty
\le
\epsilon_c^{\mathrm{abs}}
+
\epsilon_c^{\mathrm{rel}}
\max\left(
\|\mathbf C\mathbf u\|_\infty,
\|\mathbf d\|_\infty
\right).
\]

No dimensionless floor is inserted into either relative scale; the absolute tolerance handles the zero-load/zero-constraint case.

If `check_displacement_increment = true`, also require after the proposed Newton correction

\[
\|\Delta\mathbf u\|_\infty
\le
\epsilon_u^{\mathrm{abs}}
+
\epsilon_u^{\mathrm{rel}}
\max\left(
\|\mathbf u_{n+1}^{(i)}\|_\infty,
\|\mathbf u_n\|_\infty
\right).
\]

This displacement-correction test is supplementary. It must never replace equilibrium and constraint residual checks.


The corresponding TOML fields are `force_atol`, `force_rtol`, `constraint_atol`, `constraint_rtol`, `displacement_atol`, and `displacement_rtol`.

---

## 18. Pseudo-time, curves, mandatory events, and cutback

### 18.1 Curves

A scalar curve is piecewise linear through strictly increasing pseudo-time knots. Do not extrapolate outside its declared domain unless an explicit future option allows it.

Use curves for prescribed displacement amplitudes, nodal-force amplitudes, and components of prescribed macroscopic RVE deformation.

### 18.2 Mandatory event set

The solver must land exactly, to a scale-aware time tolerance, on

\[
\boxed{
\mathcal T_{\mathrm{event}}
=
\mathcal T_{\mathrm{curve\ knots}}
\cup
\mathcal T_{\mathrm{output}}
\cup
\mathcal T_{\mathrm{restart}}
\cup
\{t_{\mathrm{end}}\}.
}
\]

If the user requests output every `0.01`, those times are actual solved equilibrium endpoints. Do not step over them and interpolate afterward while calling the result a solved state.

Attempted endpoint:

```text
dt_proposed = adaptive controller proposal
next_event   = smallest event > t_n
dt_attempt   = min(dt_proposed, next_event - t_n)
```

`dt_min` bounds adaptive proposals/cutbacks but may be violated once solely to land exactly on a nearby mandatory event. If that event-landing step fails and a further cutback below `dt_min` would be required, terminate.

### 18.3 Required controller fields

The TOML `[time]` table must support at least:

```toml
dt_initial = 0.10
dt_min = 1.0e-5
dt_max = 0.20
cutback_factor = 0.5
growth_factor = 1.5
grow_if_newton_iterations_le = 4
max_attempts_per_increment = 12
```

After an accepted step, growth is allowed only by the configured policy and is capped at `dt_max`. After failure, multiply the proposed step by `cutback_factor`. Every attempted endpoint is then clipped to the next mandatory event.

### 18.4 Cutback

On recoverable failure:

1. discard trial displacement/multiplier/material state;
2. retain the committed state at `t_n` unchanged;
3. reduce the proposed step by `cutback_factor`;
4. retry from `t_n`.

Do not use cutback to hide a structurally singular KKT system caused by dependent constraints or an unremoved rigid mode.

---

## 19. Failure classification

### Fatal model/setup errors

- nonpositive or degenerate reference element mapping;
- invalid committed configuration;
- unsupported element topology;
- missing/ambiguous Physical Group assignment;
- incompatible material state layout on restart;
- dependent/inconsistent constraints detected as a structural setup error;
- unsupported non-translational periodic transformation.

### Recoverable trial failures

- current Newton trial gives invalid current element Jacobian;
- `det(F_np1) <= 0`;
- invalid F-bar projection at a trial endpoint;
- local material integration reports recoverable nonconvergence;
- global Newton fails within the iteration limit.

Recoverable failures request a cutback from the unchanged committed state.

---

## 20. Input deck

Use TOML for analysis/control data and a Gmsh `.msh` file for mesh topology/regions/periodicity. The five files under `examples/case_*.toml` are normative schema examples for the first acceptance suite.

Minimum top-level responsibilities:

```toml
[analysis]
[mesh]
[time]
[nonlinear]
[linear_solver]
[[curves]]
[[materials]]
[[element_assignments]]
[[constraints.prescribed]]
[[constraints.linear]]          # optional raw affine equation
[[constraints.periodic_rve]]    # optional
[[loads.nodal]]                 # optional dead nodal force
[output]
[restart]
[verification]
```


A scalar value expression may contain a constant term, a curve term, or both. When both are present, evaluate

\[
q(t)=q_0+s\,c(t).
\]

Thus `{constant = 1.0, curve = "ramp", scale = 0.10}` means `1.0 + 0.10*ramp(t)`.

### 20.1 Materials

Example:

```toml
[[materials]]
name = "matrix"
model = "neo_hookean"
[materials.parameters]
mu = 1.0
kappa = 20.0

[[element_assignments]]
region = "solid"
formulation = "hex8_fbar"
material = "matrix"
```

The assignment `region` must name a 3D Gmsh Physical Group.

### 20.2 Prescribed displacement

A named boundary group expands into one constraint row per selected nodal component.

```toml
[[constraints.prescribed]]
name = "clamp_xmin"
region = "xmin"
components = ["x", "y", "z"]
value = { constant = 0.0 }
```

### 20.3 Raw linear equation

Use explicit node/DOF terms for one mathematical equation. Do not overload a node-set name to mean either a sum or row generation. `node_tag` refers to the original Gmsh node tag exposed by the mesh adapter.

```toml
[[constraints.linear]]
name = "example_mpc"
terms = [
  { node_tag = 101, component = "x", coefficient = 1.0 },
  { node_tag = 205, component = "x", coefficient = -1.0 }
]
rhs = { constant = 0.0 }
```

The loader converts this into one row of `C u = d`. It must reject duplicate terms that are not intentionally combined, unknown node tags/components, and an all-zero row.

### 20.4 Periodic RVE

The input supplies the prescribed macroscopic deformation history; Gmsh supplies node pairing. The acceptance decks use:

```toml
[[constraints.periodic_rve]]
slave_regions = ["xmax", "ymax", "zmax"]
anchor_region = "anchor"
macro_F = [
  [{constant = 1.0, curve = "ramp", scale = 0.10}, {constant = 0.0}, {constant = 0.0}],
  [{constant = 0.0}, {constant = 1.0}, {constant = 0.0}],
  [{constant = 0.0}, {constant = 0.0}, {constant = 1.0}]
]
```

Each named slave surface must have Gmsh periodic metadata identifying its master. The anchor group must resolve to exactly one reference node for the supplied RVE case. Generate

\[
d_{ab}(t)=[\overline{\boldsymbol F}(t)-\boldsymbol I](\boldsymbol X_b-\boldsymbol X_a)
\]

for retained periodic edges in the constraint graph.

---

## 21. Output and restart

### 21.1 Output/restart schedule syntax

`interval` generates uniformly spaced mandatory events starting from `t_start` and not exceeding `t_end`; `explicit_times` adds additional events. Duplicate times from load curves, output, restart, or the final endpoint are merged within the controller's time tolerance.

### 21.2 Accepted-state output only

Normal state output is written only after global convergence/commit at a mandatory output event.

At minimum support:

- pseudo-time;
- nodal displacement;
- constraint reaction vector or recoverable boundary reaction totals;
- Newton iteration history;
- selected Gauss-point fields;
- material state variables when requested.

VTU is suitable for visualization output. HDF5 is suitable for restart and structured state data.

### 21.3 Restart contents

Store enough to reproduce the accepted state exactly:

```text
t_n
next/proposed dt controller state
u_n
lambda_n
committed material-state blocks
mesh/assignment/schema identity needed for compatibility checks
```

Do **not** store `F_n` as authoritative history. Reconstruct it from reference coordinates and committed `u_n`.

Cold start calls `initialize_material_state`. Restart does not; it loads committed state after compatibility checks.

---

## 22. Verification tests required during implementation

### 22.1 Mathematical/unit tests

1. Hex8/Hex20 shape-function partition of unity and derivative sums.
2. Nodal interpolation checks.
3. Reference/current Jacobian and `F = Jx @ inv(J0)` checks.
4. Rigid translation: `F = I`, zero internal force for stress-free material.
5. Homogeneous affine deformation reproduction.
6. Neo-Hookean analytic `dP/dF` versus centered finite difference.
7. Standard element directional derivative:
   \[
   \mathbf K^e\Delta\mathbf u
   \approx
   \frac{\mathbf f^{e,\mathrm{int}}(\mathbf u+\epsilon\Delta\mathbf u)-
   \mathbf f^{e,\mathrm{int}}(\mathbf u-\epsilon\Delta\mathbf u)}{2\epsilon}.
   \]
8. F-bar element directional derivative with the full projection tangent.
9. Independent F-bar reference-form versus spatial-tangent implementation check.
10. Homogeneous state: standard Hex8 and Hex8-Fbar residual/stress/state agree. Do **not** require their complete tangent matrices to be identical for arbitrary non-affine perturbations; require agreement of tangent action for homogeneous affine perturbations and require each full tangent to match its own finite-difference residual derivative.
11. KKT constraint satisfaction and global force balance.
12. Augmented global directional derivative at fixed pseudo-time. For an arbitrary test direction `(du, dlambda)`, verify
   \[
   \begin{bmatrix}\mathbf K&\mathbf C^\top\\\mathbf C&\mathbf0\end{bmatrix}
   \begin{bmatrix}\Delta\mathbf u\\\Delta\boldsymbol\lambda\end{bmatrix}
   \approx
   \frac{
   \boldsymbol{\mathcal R}(\mathbf u+\epsilon\Delta\mathbf u,
   \boldsymbol\lambda+\epsilon\Delta\boldsymbol\lambda)
   -
   \boldsymbol{\mathcal R}(\mathbf u-\epsilon\Delta\mathbf u,
   \boldsymbol\lambda-\epsilon\Delta\boldsymbol\lambda)
   }{2\epsilon},
   \]
   where
   \[
   \boldsymbol{\mathcal R}(\mathbf u,\boldsymbol\lambda)
   =
   \begin{bmatrix}
   \mathbf f^{\mathrm{int}}(\mathbf u)-\mathbf f^{\mathrm{ext}}+\mathbf C^\top\boldsymbol\lambda\\
   \mathbf C\mathbf u-\mathbf d
   \end{bmatrix}.
   \]
   Keep `C`, `d`, external loading, and pseudo-time fixed during this perturbation.
13. Restart equivalence.
14. Cutback/rollback test with deliberately forced recoverable failure.

---

## 23. Executable acceptance problems

### 23.1 Case A: elongated clamped block

Geometry:

```text
Lx x Ly x Lz = 4 x 1 x 1
```

Gmsh Physical Groups:

```text
solid, xmin, xmax, ymin, ymax, zmin, zmax
```

Baseline linear mesh: `8 x 2 x 2` Hex8 elements. Generate a compatible second-order serendipity mesh for Hex20.

Boundary conditions:

- `xmin`: `ux = uy = uz = 0`;
- `xmax`: prescribed `ux(t) = 0.4 * ramp(t)`;
- transverse components on `xmax` are free;
- other faces free.

Material: compressible neo-Hookean reference material.

Run separately with:

- `hex8`;
- `hex8_fbar`;
- `hex20`.

Required checks:

- every mandatory event converges for reference parameters without unintended cutback;
- positive `J` at every material point;
- constraint residual tolerance;
- global force balance including constraint reactions;
- selected element/global tangent directional derivatives;
- restart equivalence at `t = 0.5`.

The three discretizations are not required to have identical coarse-mesh reaction curves.

### 23.2 Case B: periodic homogeneous 8x8x8 RVE

Geometry: unit cube with exactly `8 x 8 x 8` linear Hex8 elements.

Use Gmsh translational periodic correspondence for opposite faces and one origin/reference anchor.

Prescribe

\[
\overline{\boldsymbol F}(t)
=\boldsymbol I+0.1\,ramp(t)\,\boldsymbol e_1\otimes\boldsymbol e_1.
\]

Run separately with:

- `hex8`;
- `hex8_fbar`.

For homogeneous neo-Hookean material, require the recovered displacement field to satisfy

\[
\boldsymbol u(\boldsymbol X,t)
=[\overline{\boldsymbol F}(t)-\boldsymbol I]\boldsymbol X
\]

up to the configured tolerance after the gauge translation is fixed. Every Gauss point should recover the prescribed homogeneous `F` to numerical precision.

For this homogeneous solution, standard Hex8 and Hex8-Fbar must agree in stress, internal/reaction response, and material state. Their complete tangent matrices are not required to be identical for arbitrary non-affine perturbations.

---

## 24. Development sequence

Recommended order:

1. Gmsh generators/import and local ordering validation.
2. Shape functions/quadrature/kinematics.
3. Neo-Hookean material and `dP/dF` verification.
4. One-element standard Hex8 residual/tangent.
5. Standard Hex8 Case A.
6. Sparse affine constraints and KKT solve.
7. Periodic standard Hex8 Case B.
8. Hex8-Fbar residual/tangent and its verification suite.
9. Hex20 full-integration element and Case A.
10. Time/event/cutback controls.
11. Output and restart.
12. Stateful finite-strain J2 material through the unchanged material API.
13. Only after all of the above, integrate crystal plasticity.

If a later finite-strain plasticity model uses a multiplicative split, the formulation notation is `F = F_e F_p`, with `e` and `p` as subscripts.

The agent may propose a different order, but should explain why it lowers implementation risk.

---

## 25. Handoff rule for implementation agents

The implementation agent should not need to derive the formulas in this document. It should:

- implement the stated contracts;
- use the formulation reference only to clarify mathematical meaning or audit a formula;
- build verification tests at the same time as each numerical component;
- stop and report a contradiction rather than silently changing the formulation;
- avoid weakening acceptance tolerances simply to obtain a passing run.
