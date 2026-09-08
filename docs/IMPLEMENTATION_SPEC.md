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
preprocess/        input loading, validation, mesh/model/constraint construction
assembly/          DOF numbering and sparse global assembly
solver/            Newton, KKT linear solve, increment controller
io/                append-only HDF5 results, JSON run log, HDF5 restart
postprocess/       temporal XDMF or other derived visualization formats
verification/      tangent and acceptance checks
```

No element kernel should import Gmsh. No material model should know which element formulation called it.
The solver receives a prepared analysis and never writes a visualization format. Postprocessing reads the
accepted-state database without re-running the solver.

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

$$
\boldsymbol x_{A,n}=\boldsymbol X_A+\boldsymbol u_{A,n},\qquad
\boldsymbol x_{A,n+1}^{(i)}=\boldsymbol X_A+\boldsymbol u_{A,n+1}^{(i)}.
$$

### 4.2 Deformation gradients are not history variables

**Invariant:** `F_n`, `F_np1`, raw or projected, are reconstructed from nodal kinematics whenever an element is evaluated. They are not independently advanced Gauss-point state and are not required in a restart file.

### 4.3 Constitutive data categories

Keep three categories distinct:

- **material properties** are immutable values shared by every point using one named material definition, such as `mu` and `kappa`;
- **point properties** are immutable but may vary by element or integration point, such as an orientation or a catalog `mat_id`;
- **material state** contains only evolving, history-dependent integration-point fields.

The registered material model declares its property schema and named state layout once. Individual material definitions provide property values but do not repeat those schemas. An element assignment remains separate and maps a Gmsh region to an element formulation, a material definition, and an optional point-property source.

Each model state layout has a fixed packed size before allocation. For each homogeneous element/material block, keep separate committed and trial arrays, for example

```text
state_n:     float64 [n_elem_block, n_gauss, n_state]
state_trial: float64 [n_elem_block, n_gauss, n_state]
```

`n_state = 0` is valid for hyperelasticity. In that case no material-point initializer is called. The update receives a zero-length state view so its signature remains uniform.

Only accepted global states are committed. A new Newton iteration must never use the previous Newton iteration's trial state as its starting state.

### 4.4 Named request/response records

Do not use positional tuples for material or element APIs.

Suggested Python records:

```python
@dataclass(frozen=True)
class MaterialRequest:
    F_n: np.ndarray          # (3, 3)
    F_np1: np.ndarray        # (3, 3), current trial endpoint
    state_n: MaterialStateView
    properties: object
    point_properties: object | None
    t_n: float
    t_np1: float
    need_tangent: bool

@dataclass
class MaterialResponse:
    P: np.ndarray                    # (3, 3)
    A_alg: np.ndarray | None         # (3, 3, 3, 3), dP_np1/dF_np1
    state_trial: MaterialStateView
    status: MaterialStatus
```

The order shown here is not a tuple order; access fields by name.

`need_tangent=False` may skip construction of `A_alg`, but **must not alter** `P`, `state_trial`, or `status`.

---

## 5. Material registry and material assignment

### 5.1 Orthogonal model, property, point-data, and assignment layers

Do not encode constitutive meaning in a single integer material ID.

Use four separate records:

```python
@dataclass(frozen=True)
class MaterialModel:
    root: str
    update: Callable
    initialize: Callable | None
    validate_properties: Callable
    state_layout: StateLayout

@dataclass(frozen=True)
class MaterialDefinition:
    name: str
    model: MaterialModel
    properties: Mapping[str, Any]
```

The effective material binding is the combination of:

```text
registered model x named property set x optional point-property source x element assignment
```

This is a composition of independent concerns, not a request to enumerate a literal Cartesian product. An element assignment maps a named Gmsh volume Physical Group to:

- element formulation (`hex8`, `hex8_fbar`, `hex20`);
- material definition name;
- optional property source.

For crystal plasticity later, one phase/material definition may be shared by many elements while initial orientation is supplied as an element or integration-point property. A `mat_id` may reference an immutable catalog loaded once during preprocessing. Resolve and cache that catalog before element evaluation; material-point updates must not perform repeated file I/O. Record the catalog identity in restart compatibility metadata. Do not create one constitutive model definition merely because the orientation differs.

### 5.2 Registered solver-native material routines

Each model has an identifier-safe root such as `neo_hook`. Register an update named `update_<root>` and, only when history state requires initialization, an initializer named `init_<root>`. Registration is explicit; do not use dynamic `eval`, implicit module globals, or string-based function execution.

Public interface:

```python
@dataclass(frozen=True)
class MaterialInitRequest:
    properties: object
    point_properties: object | None
    X: np.ndarray              # reference position of material point, shape (3,)
    t0: float

@dataclass
class MaterialInitResponse:
    state0: np.ndarray         # shape (n_state,)
    status: MaterialStatus

init_<root>(MaterialInitRequest) -> MaterialInitResponse       # optional
update_<root>(MaterialRequest) -> MaterialResponse             # required
```

The default cold start is the stress-free reference configuration with `u_0 = 0` and `F_0 = I`. For a history-dependent model, state is returned by its registered initializer, which may use point properties. A nonempty state layout without an initializer is a setup error. For a state-free model such as `neo_hook`, skip integration-point initialization entirely. Initial stress is zero for the required reference neo-Hookean model. A future material that supports nonzero initial stress requires an explicit extension of this contract rather than an implicit solver-side assumption.

The solver owns stress-measure and tangent transformations. A material returns first Piola-Kirchhoff stress and `dP/dF` only.

### 5.4 Standalone material-point path driver

Provide a solver-independent verification driver that accepts a registered material, either a target `3 x 3` deformation gradient or a callable path `F(s)`, and a positive number of equal path increments. It must call the material initializer once, advance from `s=0` to `s=1`, and commit each successful local state before the next local increment. This sequential local history is deliberate and is distinct from the global Newton rule, where every trial within one global increment starts from the same committed state.

The driver records `F`, `P`, Kirchhoff and Cauchy stress, Green--Lagrange, Euler--Almansi, material and spatial logarithmic strain, packed material state, and, when requested, `dP/dF` and its current-configuration Truesdell form. These are verification arrays rather than solver HDF5 fields; recording `F` here does not change the rule that production output reconstructs `F` from nodal displacement. A state-free material skips initialization and uses an empty state.

The J2 characterization utility must exercise at least 400 increments to 10% isochoric logarithmic uniaxial strain and 400 increments to `F12=0.1` simple shear, compare final values against 200-increment histories, save machine-readable arrays and CSV tables, and plot stress, equivalent plastic strain, and representative elastic/plastic tangent components.

---

## 6. Reference material: compressible neo-Hookean

This model is required before any stateful plasticity model.

Use

$$
W(\boldsymbol F)
=\frac{\mu}{2}\bigl(\operatorname{tr}\boldsymbol C-3\bigr)
-\mu\ln J
+\frac{\kappa}{2}(\ln J)^2,
\qquad
\boldsymbol C=\boldsymbol F^\top\boldsymbol F,
\qquad J=\det\boldsymbol F>0.
$$

Return

$$
\boxed{
\boldsymbol P
=\mu\left(\boldsymbol F-\boldsymbol F^{-\top}\right)
+\kappa\ln J\,\boldsymbol F^{-\top}
}
$$

and

$$
\boxed{
A^{\mathrm{alg}}_{iI jJ}
=\mu\,\delta_{ij}\delta_{IJ}
+\kappa(F^{-\top})_{iI}(F^{-\top})_{jJ}
+(\mu-\kappa\ln J)(F^{-\top})_{iJ}(F^{-\top})_{jI}.
}
$$

There is no material state: `n_state = 0`.

If `J <= 0` or the deformation is numerically invalid, return a recoverable trial failure. Do not return NaNs and continue assembly.

Unit tests must compare the analytic `A_alg` with centered finite differences of the complete `P(F)` evaluation.

---

## 6A. Optional stateful example: finite-strain J2 plasticity

The mathematical definition is isolated in Appendix A of `FORMULATION.tex`; it is not part of the generic FE formulation. Register this plug-in as:

```text
root:       j2_plasticity
initializer: init_j2_plasticity
update:      update_j2_plasticity
```

Required immutable properties are:

```text
shear_modulus
bulk_modulus
initial_yield_stress
linear_hardening_modulus
saturation_increment
saturation_rate
```

Reject missing or unknown properties. The two elastic moduli and initial yield stress must be positive; hardening parameters must be nonnegative. All values must be finite.

The packed history layout is:

```text
plastic_metric_inverse       shape (6,), order 11,22,33,12,23,13, tensorial shear
equivalent_plastic_strain    scalar
```

Initialize these fields to `[1,1,1,0,0,0]` and zero. The update must reconstruct the full symmetric tensor, reject a non-finite, non-positive-definite committed metric, and return a recoverable failure for invalid trial kinematics or local nonconvergence.

The update is the radial return and determinant-one state reconstruction specified in the formulation appendix. Solve the scalar plastic multiplier from the committed state with at most 30 Newton iterations and a scale-aware residual tolerance of `1e-12`. Enforce a nonnegative multiplier and reject a nonpositive radial stress scale. The determinant-one spherical scalar is the positive-definite root of the cubic determinant equation; bracket it and use a fixed 80 bisections so its result is deterministic.

When `need_tangent=true`, construct `A_alg` by applying the appendix's exact directional linearization to all nine Cartesian basis perturbations of `F_np1`. This is an analytic algorithmic tangent, not a finite-difference production option. `need_tangent=false` must return identical stress, state, and status.

Required material tests cover validation and initialization, hydrostatic elastic loading, plastic consistency, elastic unloading, rotational covariance, state symmetry/positive-definiteness/unit determinant, and centered finite-difference checks of all smooth elastic and plastic tangent branches. Required element tests exercise a yielded standard Hex8 and Hex8-Fbar tangent from the same committed state. Restart and cutback tests must include a nonzero plastic state.

The HDF5 result stores both state fields at element centroids. The six-component plastic metric carries the same component-order metadata as reported symmetric stress and strain. Do not store an additional J2/von-Mises stress field; it is derived from the saved Cauchy stress during postprocessing.

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

$$
\boldsymbol u^+ - \boldsymbol u^-
=
(\overline{\boldsymbol F}-\boldsymbol I)
(\boldsymbol X^+-\boldsymbol X^-).
$$

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

$$
\mathbf J_0=\frac{\partial\boldsymbol X}{\partial\boldsymbol\xi},
\qquad
\mathbf J_{x,n}=\frac{\partial\boldsymbol x_n}{\partial\boldsymbol\xi},
\qquad
\mathbf J_{x,n+1}^{(i)}=\frac{\partial\boldsymbol x_{n+1}^{(i)}}{\partial\boldsymbol\xi}.
$$

With column gradients,

$$
\nabla_X N_A=\mathbf J_0^{-\top}\nabla_\xi N_A,
\qquad
\nabla_x N_A=(\mathbf J_{x,n+1}^{(i)})^{-\top}\nabla_\xi N_A.
$$

Reconstruct endpoint deformation gradients:

$$
\boxed{
\boldsymbol F_n=\mathbf J_{x,n}\mathbf J_0^{-1},
\qquad
\boldsymbol F_{n+1}^{(i)}=\mathbf J_{x,n+1}^{(i)}\mathbf J_0^{-1}.
}
$$

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
   $$
   \boldsymbol\sigma=J^{-1}\boldsymbol P\boldsymbol F^\top.
   $$
5. If a tangent is requested, transform `A_alg = dP/dF` to the Truesdell spatial tangent using formulation labels `eq:a-pushforward` and `eq:cT-from-A`:
   $$
   a^{\mathrm{pf}}_{ij km}
   =\frac1J A^{\mathrm{alg}}_{iI kK}F_{jI}F_{mK},
   $$
   $$
   c^{\mathrm{T,alg}}_{ij km}
   =\frac12\left[
   a^{\mathrm{pf}}_{ij km}+a^{\mathrm{pf}}_{ij mk}
   -\delta_{ik}\sigma_{mj}-\delta_{im}\sigma_{kj}
   \right].
   $$
6. Build the ordinary 6-by-`3*n_e` engineering-shear `B` matrix from current gradients. Use exactly the Voigt convention in formulation label `eq:voigt-explicit`.
7. Current quadrature volume:
   $$
   \omega_g^x=\det(\mathbf J_{x,n+1}^{(i)})W_g.
   $$
8. Accumulate
   $$
   \mathbf f^{e,\mathrm{int}} \mathrel{+}=
   \mathbf B_g^\top\boldsymbol\sigma_g^{\mathrm{V}}\omega_g^x,
   $$
   $$
   \mathbf K^{e,\mathrm{mat}} \mathrel{+}=
   \mathbf B_g^\top\mathbf D_g^{\mathrm{T,alg}}\mathbf B_g\omega_g^x,
   $$
   and for node pair `(A,B)`
   $$
   \mathbf K^{e,\mathrm{geo}}_{AB}\mathrel{+}=
   \left[(\nabla_xN_A)^\top\boldsymbol\sigma_g\nabla_xN_B\right]\mathbf I_3\omega_g^x.
   $$
9. Store the returned state only in the element trial buffer.

Return

$$
\boxed{
\mathbf K^e=\mathbf K^{e,\mathrm{mat}}+\mathbf K^{e,\mathrm{geo}}.
}
$$

Do not assume symmetry and do not store only one triangle.

---

## 11. Centroidal Hex8-Fbar kernel

The material API must remain identical to Section 10.

### 11.1 Endpoint projection

At both the committed endpoint and current trial endpoint, reconstruct the raw Gauss-point and centroid deformation gradients.

For each Gauss point

$$
\alpha_g=\left(\frac{J_c}{J_g}\right)^{1/3},
\qquad
\overline{\boldsymbol F}_g=\alpha_g\boldsymbol F_g.
$$

Do this separately for `n` and `n+1,i`.

Call the material routine with

```text
F_n   = Fbar_n_g
F_np1 = Fbar_trial_g
```

The material returns the same `P`, `A_alg`, state, and status fields as any other element. For this element denote them mathematically by `Pbar` and `Abar` only to make the element derivation readable; the software response type is unchanged.

### 11.2 Stress and residual

At a Gauss point

$$
\overline{\boldsymbol\sigma}_g
=J_c^{-1}\overline{\boldsymbol P}_g\overline{\boldsymbol F}_g^\top.
$$

Use **raw current geometry** to build `B`, `grad_x N`, and the current quadrature weight `omega_x[g] = omega_g^x`. Accumulate

$$
\mathbf f^{e,\mathrm{int}}
\mathrel{+}=
\mathbf B_g^\top\overline{\boldsymbol\sigma}_g^{\mathrm V}\omega_g^x.
$$

### 11.3 Consistent spatial tangent

Push `Abar` forward with `Fbar` and `J_c` to obtain the material Truesdell tangent `Dbar_T_alg` using the same transformation as Section 10.

For each node `B`, define

$$
\boldsymbol q_{B,g}
=\frac13\left(\nabla_x N_B|_c-\nabla_x N_B|_g\right).
$$

For component `k = 0,1,2`, define the projected column of the node block:

$$
\overline{\mathbf B}^{\,g}_B[:,k]
=
\mathbf B^{\,g}_B[:,k]
+q_{B,g,k}
\begin{bmatrix}1&1&1&0&0&0\end{bmatrix}^{\!\top}.
$$

This notation is deliberately explicit: `B_B^g` is a 6-by-3 node block; `[:,k]` is programming-array notation used only in this implementation document.

Accumulate

$$
\mathbf K^{e,\mathrm{mat}}_{\bar F}
\mathrel{+}=
\mathbf B_g^\top
\overline{\mathbf D}^{\mathrm{T,alg}}_g
\overline{\mathbf B}_g\,\omega_g^x,
$$

ordinary geometric blocks using `sigma_bar`, and

$$
\boxed{
\mathbf K^{e,\mathrm{proj}}_{AB}
\mathrel{+}=
-\left(\overline{\boldsymbol\sigma}_g\nabla_x N_A|_g\right)
\otimes\boldsymbol q_{B,g}\,\omega_g^x.
}
$$

Return

$$
\boxed{
\mathbf K^e_{\bar F}
=
\mathbf K^{e,\mathrm{mat}}_{\bar F}
+\mathbf K^{e,\mathrm{geo}}_{\bar F}
+\mathbf K^{e,\mathrm{proj}}_{\bar F}.
}
$$

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

$$
\mathbf f^{e,\mathrm{body}}_A
=\int_{\Omega_{0e}}N_A\boldsymbol b_0\,\mathrm dV
\approx\sum_g N_A(\boldsymbol\xi_g)\boldsymbol b_0(\boldsymbol X_g,t)
\det\mathbf J_{0,g}W_g.
$$

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

### 13.1 Local process parallelism

Element kernels may execute in a local process pool. The unit of submitted work is one complete element, not one
quadrature point: the worker gathers all quadrature-point material updates, the element residual, the element tangent,
and the trial state into one element response. Consequently, the Hex8-Fbar centroid quantities and projection chain
rule remain private to one element invocation and use the same material-point API as the serial path.

Create a fixed pool once per analysis and reuse it for initial output, every Newton and line-search assembly, and final
verification. `--num-processes N` is a command-line execution choice with default `N=1`; it is not part of the TOML
model definition or restart identity. Never select all available CPUs implicitly. Report the requested and effective
process counts and the available physical/logical CPU counts so the user can choose an appropriate value. The effective
count may be reduced to the number of elements. Limit numerical-library thread pools to one thread inside each element
worker to avoid nested oversubscription; global sparse-solver threading is outside this v1 parallelization.

Workers receive serializable element-local copies and return element responses. They do not mutate the mesh, global
vectors/matrices, committed material state, result database, or JSON log. The parent consumes responses in the original
block/element order, adds the element force to the dense global vector, and appends stiffness COO triplets in that same
order. Workers may finish out of order, but deterministic parent-side reduction preserves the serial floating-point
summation order and avoids solver-path changes caused only by completion timing. A recoverable element failure must
leave committed state unchanged and the persistent pool usable by the next line-search candidate or cutback attempt.

Use `forkserver` where available and otherwise `spawn`; material routines used with multiple processes must therefore
be importable and material properties must be serializable. Keep the single-process path available for reference runs,
debugging, and models for which process transfer overhead exceeds element-kernel work.

---

## 14. General affine displacement constraints

Represent all displacement constraints as

$$
\boxed{\mathbf C\mathbf u=\mathbf d(t)}
$$

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

$$
\mathbf r_u
=\mathbf f^{\mathrm{int}}-\mathbf f^{\mathrm{ext}}
+\mathbf C^\top\boldsymbol\lambda,
\qquad
\mathbf r_c=\mathbf C\mathbf u-\mathbf d.
$$

Solve the complete augmented system

$$
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
$$

The KKT matrix is expected to be indefinite. Do not treat indefiniteness as singularity.

For full-row-rank `C`, local constrained solvability requires the reduced tangent `Z.T @ K @ Z` to be nonsingular, where columns of `Z` span `null(C)`. This is a diagnostic statement; v1 does not need to construct `Z` in the production solve.

Physical constraint reactions are

$$
\boxed{\mathbf f_c=-\mathbf C^\top\boldsymbol\lambda.}
$$

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

### 16.3 Optional Newton backtracking

The baseline remains the undamped update. When `line_search = "backtracking"`, treat the KKT solution as a search direction and test

```text
u_candidate      = u_i      + alpha * delta_u
lambda_candidate = lambda_i + alpha * delta_lambda
alpha            = 1, rho, rho^2, ...
```

Every candidate assembly must integrate its material response from the unchanged committed state at `n`. A recoverable error such as an invalid current element Jacobian rejects that candidate and reduces `alpha`; it does not immediately cut back pseudo-time. For an admissible candidate, use the merit

$$
M_i=\max\left(\frac{\|\boldsymbol r_u\|_\infty}{\epsilon_{f,i}},
               \frac{\|\boldsymbol r_c\|_\infty}{\epsilon_{c,i}}\right),
$$

where the two denominators are the convergence tolerances evaluated at the base iterate. Accept the first candidate satisfying

$$
M(\alpha)\le (1-c\alpha)M_i.
$$

If no admissible residual-reducing candidate exists at or above `line_search_min_alpha`, report a recoverable global failure and let the normal increment cutback logic restart from the committed state. For exact Newton, a candidate assembly may compute and retain its tangent for the next iteration; this is still exact Newton because the retained tangent belongs to the accepted candidate iterate. Log the accepted `alpha`, number of trials/backtracks, candidate merit, count of recoverable candidate rejections, and the last such failure message in the standalone JSON history.

Supported controls and defaults are:

```toml
line_search = "none"              # or "backtracking"
line_search_reduction = 0.5
line_search_armijo = 1.0e-4
line_search_min_alpha = 1.0e-4
line_search_max_backtracks = 14
```

### 16.4 State commit

At iteration `i`, always integrate

$$
(\boldsymbol F_n,\mathcal H_n)
\rightarrow
(\boldsymbol F_{n+1}^{(i)},\widehat{\mathcal H}_{n+1}^{(i)}).
$$

Never integrate from trial iteration `i` to trial iteration `i+1`.

On global convergence, commit all Gauss-point trial states atomically. On failure, discard all trial state.

---

## 17. Convergence controls

Use explicit absolute-plus-relative tests. A reference form is

$$
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
$$

and

$$
\|\mathbf r_c\|_\infty
\le
\epsilon_c^{\mathrm{abs}}
+
\epsilon_c^{\mathrm{rel}}
\max\left(
\|\mathbf C\mathbf u\|_\infty,
\|\mathbf d\|_\infty
\right).
$$

No dimensionless floor is inserted into either relative scale; the absolute tolerance handles the zero-load/zero-constraint case.

If `check_displacement_increment = true`, also require after the proposed Newton correction

$$
\|\Delta\mathbf u\|_\infty
\le
\epsilon_u^{\mathrm{abs}}
+
\epsilon_u^{\mathrm{rel}}
\max\left(
\|\mathbf u_{n+1}^{(i)}\|_\infty,
\|\mathbf u_n\|_\infty
\right).
$$

This displacement-correction test is supplementary. It must never replace equilibrium and constraint residual checks.


The corresponding TOML fields are `force_atol`, `force_rtol`, `constraint_atol`, `constraint_rtol`, `displacement_atol`, and `displacement_rtol`.

---

## 18. Pseudo-time, curves, mandatory events, and cutback

### 18.1 Curves

A scalar curve is piecewise linear through strictly increasing pseudo-time knots. Do not extrapolate outside its declared domain unless an explicit future option allows it.

Use curves for prescribed displacement amplitudes, nodal-force amplitudes, and components of prescribed macroscopic RVE deformation.

### 18.2 Mandatory event set

The solver must land exactly, to a scale-aware time tolerance, on

$$
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
$$

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

Use TOML for analysis/control data and a Gmsh `.msh` file for mesh topology/regions/periodicity. The checked-in TOML
decks under `examples/` are normative schema examples for the acceptance suite.

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
[[constraints.affine]]          # optional full affine boundary motion
[[constraints.periodic_rve]]    # optional
[[loads.nodal]]                 # optional dead nodal force
[output]
[restart]
[verification]
```


A scalar value expression may contain a constant term, a curve term, or both. When both are present, evaluate

$$
q(t)=q_0+s\,c(t).
$$

Thus `{constant = 1.0, curve = "ramp", scale = 0.10}` means `1.0 + 0.10*ramp(t)`.

### 20.1 Materials

Example:

```toml
[[materials]]
name = "matrix"
model = "neo_hook"
[materials.properties]
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

The input supplies the prescribed macroscopic deformation history; Gmsh supplies node pairing. An explicit matrix is:

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

$$
d_{ab}(t)=[\overline{\boldsymbol F}(t)-\boldsymbol I](\boldsymbol X_b-\boldsymbol X_a)
$$

for retained periodic edges in the constraint graph.

Exactly one of `macro_F` and `macro_deformation` is required. Named exact paths avoid approximating a nonlinear
kinematic relation between curve knots:

```toml
macro_deformation = {type = "isochoric_uniaxial", axis = "x", stretch = {constant = 1.0, curve = "ramp", scale = 0.2}}
```

This gives `F11=lambda` and the two transverse stretches `lambda^(-1/2)` at every trial time, including a cutback
endpoint. The other required path is

```toml
macro_deformation = {type = "simple_shear", direction = "x", normal = "y", amount = {curve = "ramp", scale = 0.2}}
```

which gives `F12=0.2*ramp(t)` with unit diagonal.

### 20.5 Full affine boundary motion

`constraints.affine` takes a union of boundary Physical Groups, an origin, and the same `macro_F` or
`macro_deformation` description used by periodic constraints. Deduplicate nodes shared by several faces and generate
one row per selected node and Cartesian component:

```toml
[[constraints.affine]]
regions = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
origin = [0.0, 0.0, 0.0]
macro_F = [ ... ]
```

The prescribed value is `u(X,t) = [macro_F(t)-I] [X-origin]`. Only the listed groups are constrained; unlisted
boundary faces are traction-free unless separately loaded.

For the neo-Hookean uniaxial-stress rotation benchmark, use:

```toml
regions = ["xmin", "xmax"]
macro_deformation = {type = "neo_hook_uniaxial_rotation", material = "bar", stretch = {curve = "axial_stretch"}, angle_degrees = {curve = "rotation_degrees"}}
```

This exact path is restricted to axial stretch along x and rotation about z. Resolve `mu` and `kappa` from the named
`neo_hook` material; require positive finite parameters and a positive finite stretch. At each requested time,
evaluate `lambda` and `theta` from the scalar expressions. Solve
`mu*expm1(2*q) + kappa*(log(lambda)+2*q) = 0` for `q=log(a)` between `0` and `-log(lambda)/2`
(sort the endpoints; use `q=0` for `lambda=1`). Return `Rz(theta) @ diag(lambda,a,a)` with `a=exp(q)`.
Use degree-to-radian conversion for `angle_degrees`. Evaluate the path exactly at cutback endpoints too; do not
linearly interpolate deformation-matrix entries. Cache repeated evaluations at a common time if useful.
This is a benchmark boundary prescription, not a constitutive routine; the element/material interface is unchanged.

---

## 21. Output and restart

### 21.1 Output/restart schedule syntax

`interval` generates uniformly spaced mandatory events starting from `t_start` and not exceeding `t_end`; `explicit_times` adds additional events. Duplicate times from load curves, output, restart, or the final endpoint are merged within the controller's time tolerance.

### 21.2 Accepted-state HDF5 database

The solver creates one HDF5 result database per run. Append the cold-start or resumed accepted state and every
subsequent globally converged, committed increment. A mandatory output time is therefore a solved database row;
accepted endpoints introduced by cutback or growth control are retained as well. Never write a failed Newton trial.

The required schema is logically:

```text
/meta/resolved_input_json
/mesh/reference_coordinates                   [n_node, 3]
/mesh/node_tags                               [n_node]
/mesh/blocks/<block>/connectivity             [n_elem, n_node_per_elem]
/mesh/blocks/<block>/element_tags             [n_elem]
/materials/<material>                         model/properties/state-layout metadata
/curves/<curve>/{time,value}
/results/time                                 [n_step]
/results/nodal/displacement                   [n_step, n_node, 3]
/results/nodal/constraint_reaction            [n_step, n_node, 3]
/results/blocks/<block>/cauchy_stress         [n_step, n_elem, 6]
/results/blocks/<block>/green_lagrange_strain [n_step, n_elem, 6]
/results/blocks/<block>/euler_almansi_strain  [n_step, n_elem, 6]
/results/blocks/<block>/state/<field>         [n_step, n_elem, *field_shape]
```

Block metadata identifies its region, material definition, and formulation; material metadata stores the immutable
properties used in the run. This permits region/material selection without repeating constant identifiers for every
cell. Store curves so reported fields can be correlated with prescribed histories. Store nodal constraint reactions as
`-C.T @ lambda` (the structure-on-constraint sign) because they support equilibrium audits, boundary resultants, and
later RVE homogenization. Newton residuals,
tolerances, cutback attempts, and verification summaries belong in the standalone JSON run log, not the field database.

The JSON log also records execution and performance telemetry. Its top-level `execution` object contains the element
backend, requested/effective process counts, available physical/logical CPU counts, process start method, and worker
BLAS thread limit. `timing.elapsed_wall_seconds` measures the run through the latest durable log update. Every ordinary
Newton record contains `iteration_wall_seconds`, `assembly_wall_seconds`, `line_search_assembly_wall_seconds`,
`kkt_factorization_wall_seconds`, and `kkt_solve_wall_seconds`. Line-search assembly time is the sum over attempted
candidates; when an accepted candidate assembly is reused by the next exact-Newton iteration, that next record marks
`assembly_reused_from_line_search=true` and reports zero new primary assembly time.

The optional `--debug-timing` switch additionally records `element_phase_wall_seconds` and
`sparse_finalize_wall_seconds`. The former starts after element work items have been gathered and includes element
evaluation/IPC plus ordered parent reduction; the latter covers COO construction, duplicate summation, and CSR
conversion. When backtracking is active, the analogous `line_search_element_phase_wall_seconds` and
`line_search_sparse_finalize_wall_seconds` are accumulated across successful candidate assemblies. These timers are
diagnostic wall-clock observations, not reproducible numerical results or convergence criteria.

Result schema version 3 stores symmetric stress and strain in the order `[11,22,33,12,23,13]`, with **tensorial**
shear entries (no factor of two). Store dataset attributes `component_order` (six strings), `shear_convention="tensorial"`,
and `shear_scale=1.0`. Internal element tensors and engineering-shear assembly vectors are unchanged; pack only at
the output boundary. Generic state fields retain their declared shape; do not assume that a future 3-by-3 state field
is symmetric merely from its shape. Reject version-2 result databases on resume/postprocessing with a schema diagnostic;
rerun to produce the new format. Restart schema remains version 2 because its state/kinematic contents are unchanged.

Do not store current coordinates, deformation gradients, `J`, or algorithmic material tangents. Current coordinates
and `F` are derived from reference coordinates and displacement. The tangent is an iteration-local linearization and
has no required v1 postprocessing use case.

Cell fields are reported at the parent-element centroid without adding a constitutive point:

- reconstruct the centroid kinematic `F` from reference coordinates and nodal displacement, then compute
  Green--Lagrange and Euler--Almansi strain;
- for Hex8 and Hex8-Fbar, interpolate the eight integration-point stresses and reportable state values to the center,
  which is their equal-weight arithmetic mean;
- for Hex20, use the existing center integration point `(0,0,0)` from the 3x3x3 rule;
- never call the material update at the centroid merely for output.

For F-bar this recovery does not alter or bypass the formulation: stress comes from the projected material evaluations,
while strain is explicitly a kinematic centroid quantity reconstructed from `u`. Future state variables that cannot be
meaningfully interpolated must declare a model-specific reporting operation before being added.

Make append completion transactional at the schema level: update `n_complete_steps` only after all datasets for a row
have been flushed. On resume, reject a damaged committed prefix and truncate any longer incomplete tail to that count.

### 21.3 Visualization postprocessing

A separate command reads the HDF5 database and writes one temporal XDMF entry point containing all accepted states.
Write a sidecar containing connectivity reordered for XDMF/VTK and HDF5 virtual-dataset views of the saved fields. It must not
modify constitutive results or require one visualization file per time step.

Export the symmetric fields as six-component XDMF `Matrix` attributes, preserving the database ordering rather than
using a reader-specific `Tensor6` convention. Do not add separate scalar aliases for the six components: ParaView exposes
the components of each `Matrix` attribute in its component selector, and aliases would duplicate the coloring choices.
Nodal displacement stays a three-component `Vector` on reference geometry for Warp By Vector.

Do not use XML `HyperSlab` DataItems: the ParaView XDMF3 reader can treat these as untyped arrays and drop all fields,
including displacement. Instead expose each accepted time slice as an HDF5 virtual
dataset in the sidecar, and reference it with an ordinary, explicitly typed HDF DataItem. Virtual datasets must reference
the original database using paths relative to the sidecar; they store selection metadata, not copies of field values.
Keep the original database with the visualization artifacts. Generate views only for the committed prefix.
Verify actual reader arrays and Warp By Vector through `verification/check_paraview.py` when ParaView is available;
XML parsing alone is not an interoperability test.

### 21.4 Restart contents

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

Cold start calls the registered initializer only for models with nonempty history state. Restart never initializes material points; it loads committed state after compatibility checks.
Restart is a separate HDF5 artifact from the results database. A restart does not depend on XDMF or other
postprocessing output.

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
   $$
   \mathbf K^e\Delta\mathbf u
   \approx
   \frac{\mathbf f^{e,\mathrm{int}}(\mathbf u+\epsilon\Delta\mathbf u)-
   \mathbf f^{e,\mathrm{int}}(\mathbf u-\epsilon\Delta\mathbf u)}{2\epsilon}.
   $$
8. F-bar element directional derivative with the full projection tangent.
9. Independent F-bar reference-form versus spatial-tangent implementation check.
10. Homogeneous state: standard Hex8 and Hex8-Fbar residual/stress/state agree. Do **not** require their complete tangent matrices to be identical for arbitrary non-affine perturbations; require agreement of tangent action for homogeneous affine perturbations and require each full tangent to match its own finite-difference residual derivative.
11. KKT constraint satisfaction and global force balance.
12. Augmented global directional derivative at fixed pseudo-time. For an arbitrary test direction `(du, dlambda)`, verify
   $$
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
   $$
   where
   $$
   \boldsymbol{\mathcal R}(\mathbf u,\boldsymbol\lambda)
   =
   \begin{bmatrix}
   \mathbf f^{\mathrm{int}}(\mathbf u)-\mathbf f^{\mathrm{ext}}+\mathbf C^\top\boldsymbol\lambda\\
   \mathbf C\mathbf u-\mathbf d
   \end{bmatrix}.
   $$
   Keep `C`, `d`, external loading, and pseudo-time fixed during this perturbation.
13. Restart equivalence.
14. Cutback/rollback test with deliberately forced recoverable failure.
15. For Hex8, Hex8-Fbar, Hex20, and a stateful J2/F-bar update, compare serial and two-process element responses,
    assembled residual/tangent, and trial state to machine precision. Exercise more than one assembly through the same
    pool and verify that a recoverable failure leaves committed state unchanged and does not poison the pool.

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

$$
\overline{\boldsymbol F}(t)
=\boldsymbol I+0.1\,ramp(t)\,\boldsymbol e_1\otimes\boldsymbol e_1.
$$

Run separately with:

- `hex8`;
- `hex8_fbar`.

For homogeneous neo-Hookean material, require the recovered displacement field to satisfy

$$
\boldsymbol u(\boldsymbol X,t)
=[\overline{\boldsymbol F}(t)-\boldsymbol I]\boldsymbol X
$$

up to the configured tolerance after the gauge translation is fixed. Every Gauss point should recover the prescribed homogeneous `F` to numerical precision.

For this homogeneous solution, standard Hex8 and Hex8-Fbar must agree in stress, internal/reaction response, and material state. Their complete tangent matrices are not required to be identical for arbitrary non-affine perturbations.

Cases A and B remain compact regression cases for element technology, tangent, restart, and homogeneous periodic
reproduction. The following cases are the featured physical acceptance examples.

### 23.3 Case C: stretch followed by a large rigid rotation

Use the `4 x 1 x 1` linear-Hex8 bar. Prescribe affine motion only on `xmin` and `xmax`; leave all four lateral faces
traction-free. End faces contract with the exact `neo_hook_uniaxial_rotation` path from Section 20.5, avoiding
clamp-induced transverse stress. From
`t=0` to `t=0.1`, increase the axial stretch from 1 to 1.1. Then hold that stretch and left-multiply the deformation
gradient by successive 5-degree rotations about the z axis until the total rotation is 90 degrees at `t=1`:

$$
\boldsymbol F(t=0.1)=\operatorname{diag}(1.1,a,a),
\qquad
\boldsymbol F(t=1)=\boldsymbol R_z(90^\circ)\boldsymbol F(t=0.1).
$$

For `mu=1`, `kappa=20`, the lateral stretch is approximately `a=0.95553739`. Compute it from the prescribed equation,
not this rounded value. Rotation is continuous and exact between the 5-degree output events.

Use the compressible neo-Hookean model so no constitutive history or rate-integration parameters obscure the frame
test. Compare the saved state at the end of stretch, not the undeformed state, with the final rotated state. Require:

- affine displacement reproduction and positive `J` throughout;
- `P(RF)=R P(F)` in the material unit test;
- final Cauchy stress and Euler--Almansi strain equal the 90-degree rotations of their `t=0.1` values;
- Green--Lagrange strain at `t=1` equal its `t=0.1` material-frame value;
- the dominant spatial normal components move from `11` to `22`.
- transverse stress is zero at the end of stretch; final `sigma11` and `sigma33` are zero within `1e-10`;
- Cauchy stress obeys the rotation relation at every saved rotation time, not just the final endpoint.

### 23.4 Cases D1/D2: heterogeneous periodic core--matrix cube

Use a conforming unit-cube mesh with exactly `8 x 8 x 8` linear Hex8 elements. The centered
`[0.25,0.75]^3` region contains `4 x 4 x 4 = 64` core elements; the remaining 448 elements form the matrix. Preserve
Gmsh translational node maps on all opposite faces and anchor the origin. Assign two material definitions to the two
regions. Both definitions call `neo_hook`; the core uses `mu` and `kappa` values ten times the matrix values.

Run the same Hex8-Fbar domain with two exact macroscopic paths:

$$
\overline{\boldsymbol F}_{\mathrm{D1}}(1)
=\operatorname{diag}\left(1.2,1/\sqrt{1.2},1/\sqrt{1.2}\right),
$$

and

$$
\overline{\boldsymbol F}_{\mathrm{D2}}(1)
=\boldsymbol I+0.2\,\boldsymbol e_1\otimes\boldsymbol e_2.
$$

For both cases require positive material-point `J`, global equilibrium, periodic constraint satisfaction, and
volume-average `F` equal to the prescribed macro deformation. Also require a resolved nonzero displacement
fluctuation from the affine field and a nonzero difference between the core and matrix mean stresses. The HDF5 output
must retain the two blocks/material definitions so these fields can be selected separately in postprocessing.

### 23.5 Case E: finite-strain J2 circular-bar necking

This is a system-level benchmark for the optional Appendix-A material plug-in, not a replacement for material-point verification. Use a one-eighth model: a quarter circular cross-section over half the specimen length. With full length `L = 53.334 mm`, end radius `R = 6.413 mm`, and axial coordinate `0 <= z <= L/2` measured from the middle plane, generate

$$
r(z)=R\left(0.982+0.036z/L\right).
$$

The initial middle radius is therefore `6.297566 mm`. Generate a structured all-Hex8 mesh with a three-block quarter-disk topology: a regular central square and two outer transfinite sectors meeting along the 45-degree radial line. This avoids both collapsed axis elements and the highly skewed 45-degree surface cells produced when two edges of one square block are mapped onto the same circular boundary. The baseline cross-section uses four divisions along each central-square edge and three divisions through each outer sector, giving 40 quadrilaterals. Extrude these through 24 axial layers, giving 960 Hex8 elements. Put 12 axial layers in `0 <= z <= L/6` and 12 in `L/6 <= z <= L/2`. The generator must expose Physical Groups

```text
solid, symmetry_x, symmetry_y, midplane, loaded_end, outer_surface, neck_monitor
```

Use `hex8_fbar`. Prescribe `ux=0` on `symmetry_x`, `uy=0` on `symmetry_y`, `uz=0` on `midplane`, and `uz=7 mm` on `loaded_end`; leave transverse end displacement and the outer surface free. Seven millimetres on the half-model represents 14 mm total end-to-end elongation of the mirrored specimen.

Use the millimetre--newton--MPa property set:

```text
shear_modulus = 80193.8
bulk_modulus = 164210.0
initial_yield_stress = 450.0
linear_hardening_modulus = 129.24
saturation_increment = 265.0
saturation_rate = 16.93
```

Use 50 base load increments and save exact states at end elongations 1 through 7 mm. Required checks are positive material-point Jacobians, constraint satisfaction, force balance, plastic localization at the middle, and a force--elongation curve that resolves the load maximum. Derive total end reaction, current middle radius, and the radial displacement of `neck_monitor` from stored nodal fields; do not duplicate them as solver-native datasets.

The published target radial displacement at 7 mm half-model elongation is approximately `-3.740 mm`; the cited ANSYS 3D discretization reports approximately `-3.801 mm`. Record both as external references. Because that model uses a reduced-integration mixed element rather than this solver's F-bar element, require a two-level mesh-convergence study before adopting a numerical tolerance. The helper `verification/check_j2_necking.py` reports the complete observable history and enables an explicit reference-tolerance check when requested.

### 23.6 Small J2 square-prism diagnostic

Maintain a cheap qualitative companion to Case E for nonlinear-solver diagnosis. It uses the same half-length, material, axial grading, 1.8% linear middle imperfection, symmetry conditions, and 7 mm end displacement, but replaces the quarter circle by a quarter square with two elements in each transverse direction and 24 axial layers: 96 Hex8-Fbar elements total. Choose the end half-width `R sqrt(pi)/2`, so the complete square has the same end area as the reference circle. This case is not a substitute for the circular benchmark and has no published displacement target; its purposes are to reproduce plastic localization and exercise Newton globalization quickly.

Use residual-based backtracking for both J2 necking decks. The diagnostic utilities must support (a) material and element tangent checks using evolved Gauss-point states from restart/output data and (b) dense null-space/SVD inspection of the reduced tangent only for this deliberately small model. Dense matrices remain prohibited in the production solver.

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
