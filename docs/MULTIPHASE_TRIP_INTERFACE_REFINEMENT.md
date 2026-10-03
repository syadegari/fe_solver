# Multiphase TRIP interface refinements

Status: adapter cleanup, solver-owned committed-state observation, and runtime Euler angles are implemented.
This document records the audit and decisions; executable requirements remain in `IMPLEMENTATION_SPEC.md`.

## Implemented adapter cleanup

- Initialization validates phase and the Euler-angle vector even though orientation is unused by `SDVINI`.
  A comment explains this deliberate validation.
- Library loading is already cached per process; hashing is not repeated per quadrature-point update.
- The common `need_tangent` flag stops at the Python TRIP wrapper. The legacy kernel always calculates its tangent,
  so ABI version 3 removes the unused argument. The wrapper still returns `A_alg=None` when not requested.
- J2 retains its flag because its numeric kernel actually skips tangent construction.
- Restart restores full quadrature-point history into `model.blocks` in place. A solver comment explains why only
  primary variables appear in the returned tuple and why independent stress history, if needed by a material, belongs
  in that material's declared state.

## Implemented: runtime Euler angles, two phase definitions

Keep the current two material definitions and the existing point-property source. No new grain registry, property
inheritance, or material-model hierarchy is needed. The relevant TOML remains:

```toml
[[materials]]
name = "ferrite"
model = "multiphase_trip_ferrite"
[materials.properties]
library = "libmultiphase_trip.so"

[[materials]]
name = "austenite"
model = "multiphase_trip_austenite"
[materials.properties]
library = "libmultiphase_trip.so"

[[point_property_sources]]
name = "microstructure"
file = "microstructure_properties.h5"
format = "hdf5_element_properties"

[[element_assignments]]
region = "ferrite"
formulation = "hex8_fbar"
material = "ferrite"
point_properties = "microstructure"

[[element_assignments]]
region = "austenite"
formulation = "hex8_fbar"
material = "austenite"
point_properties = "microstructure"
```

The change is in the generated property data and material ABI, not necessarily the TOML:

```text
/element_tags                    [n_element]
/properties/phase_id             [n_element]
/properties/grain_id             [n_element]
/properties/orientation_id       [n_element]       optional diagnostic identifier
/properties/euler_angles         [n_element, 3]    float64, radians
```

Flow from source data to UMAT:

1. The preprocessor reads the three angle assignments from each orientation-file branch. Those files become input
   data only, not compiler includes. It preserves the existing grain-to-phase-local-entry association and parses
   decimal/Fortran D-exponent values directly into float64 without an intermediate float32 conversion.
2. Centroid classification assigns an element to `(phase_id, grain_id)`. That pair selects its three initial angles.
3. The preprocessor writes those angles as a float64 vector keyed by the retained Gmsh element tag.
4. `build_model` uses the existing generic source loader and attaches that vector to each element's `PointProperties`.
5. Initialization and each quadrature-point update receive the element's same immutable mapping, including in workers.
6. The Python phase wrapper reads `request.point_properties["euler_angles"]`, validates a finite vector of shape
   `(3,)`, and passes its contiguous float64 buffer through the C ABI.
7. The bridge supplies `props(1:3)` with `nprops=3`. UMAT and its phase routines pass those values down to the material
   rotation routines instead of performing an orientation-ID lookup.

Keep the existing `ROTATIONMATRIX` convention exactly: radians, `RM = R3 R2 R1` with the current matrices/signs. These
are immutable initial grain orientations; evolving lattice/constitutive quantities remain internal state.

A single compiled library serves all domains using these same constitutive constants. ABI version 4 removes
orientation counts and domain/catalog-specific compilation from the bridge and builder. ABI version/state-size checks,
phase dispatch, library identity, and input shape/finiteness checks remain: those are unrelated to catalog lookup.
Existing property files need regeneration to include `euler_angles`, and older libraries need rebuilding. The TOML
structure remains unchanged. Old restart identities are not interchangeable with the new library/property identity.

### Considered alternative (not implemented): grain-specific TOML material definitions

If angles should be directly visible in TOML, they belong to each grain's immutable material definition:

```toml
[[materials]]
name = "ferrite_grain_9"
model = "multiphase_trip_ferrite"
[materials.properties]
library = "libmultiphase_trip.so"
euler_angles = [1.36554518208, 2.32289131775, 1.29310366028]

[[element_assignments]]
region = "ferrite_grain_9"
formulation = "hex8_fbar"
material = "ferrite_grain_9"
```

Here the generator creates per-grain Physical Groups, and the wrapper reads `request.properties["euler_angles"]`.
The remaining flow through the ABI is identical. This needs more definitions and assignments, plus a separate phase
grouping decision for visualization: currently each assignment produces its own output block. It does not duplicate
the constitutive model itself. The selected implementation uses the existing point-property source only; this
alternative is recorded for context, not as a supported deck format.

### Runtime-angle verification

Catalog-based ABI-3 libraries were built from companion checkpoint `fcf6b9c` before replacing the interface.
The same ABI-4 library was then used with the 8A/56F, 1A/7F, and 12A/88F inputs:

- All 172 source orientation triples were checked in a small elastic mixed-deformation update.
- Each dataset/phase's first orientation was also checked through 50 sequential shear updates to `F12=0.02`
  and 50 isochoric tensile updates to stretch `1.01`, including evolved constitutive state.
- At identical effective angles, all 772 endpoint comparisons returned bitwise-identical `P`, `dP/dF`, and state.
  This checks orientation plumbing against the unchanged integration algorithm, not the general consistency of
  its active-topology tangent through arbitrary branch changes.
- Generated 4³ property files for all three domains retained exact float64 source triples for each classified
  `(phase_id,grain_id)`, with phase element counts respectively 57/7, 55/9, and 54/10 (ferrite/austenite).

The old catalogs use unqualified decimal literals: gfortran rounds these to default real before assigning the
double-precision angle variables. The bitwise comparison therefore supplied that old effective rounding explicitly
to ABI-4. Production preprocessing instead retains the source decimals directly as float64, as agreed. In the 172
small elastic updates, this precision correction changed individual `P`, `dP/dF`, and state components by at most
`3.47e-9`, `1.64e-5`, and `9.50e-11`, respectively, in the kernel's units. These are observed absolute differences
for that check, not general error tolerances or bounds on long nonlinear trajectories.

Maintained tests cover float64/D-exponent parsing, malformed angle rejection, exact angle-buffer transfer,
orientation-ID independence, process routing, and orientation-dependent responses for both phases. The solver
suite passed 71 tests with one existing unavailable-SuperLU_MT skip; the companion suite passed six tests.

With source-precision runtime angles, the two-phase 8A/56F 4³ periodic smoke solve accepted two increments of
`dt=0.001` to `t=0.002` (`F12=0.0004`). Its raw kinematic average-`F` error was `1.56e-15`, and its force-balance
residual was `1.51e-10`. Cold-start stress was zero. Both phases also passed the one-element cold/resumed comparison
described below with the runtime-angle library. These bounded checks do not replace later finite-load TRIP studies.

## Zero-duration audit

### Static findings

Before this refinement, the solver assembled initial output using `F_n=F_np1` and equal endpoint times; final
verification also reintegrated at equal times. The standalone material-point driver evaluated its first row at time
zero. The bridge rejected these calls before UMAT. Those observation calls have now been removed.

Removing that guard is not sufficient:

- Both phase UMATs compute `SubFraction = SubFraction + DeltaT/dtime`. At zero duration this divides zero by zero,
  even when no slip system is active.
- Both return-mapping routines contain `DeltaGammaA/(GammaDot0*DeltaT)` in their slip residuals and perturbed
  residuals. Active systems therefore introduce another zero-duration singularity.
- Numerical consistent-tangent routines call those same return-mapping routines with the substep duration.
- Some heat/power divisions are guarded by `dtime /= 0`, but this does not protect the preceding expressions.

Equal deformation gradients are not themselves invalid: positive elapsed time at fixed deformation can still evolve
the rate-dependent internal state. Do not conflate `F_n=F_np1` with `dt=0`, and do not substitute a tiny positive time
increment to obtain output.

### Bounded numerical probe

An isolated Fortran driver called the actual UMAT directly, deliberately bypassing bridge validation. The executable
enabled `-ffpe-trap=invalid,zero,overflow`; the constitutive library and physical calculations were unchanged. Each
case ran in a separate subprocess with a 30-second limit, using phase-local orientation entry 1 of the 8A/56F catalog.

The evolved state was obtained through 50 accepted material-point updates with `dt=0.01` and prescribed
`F12=0.0004*step` (all diagonal entries one). Both phases ended with four active slip systems. The probe then held the
deformation gradient fixed and tested either zero duration or `dt=0.01`.

| Phase | Initial state, dt=0 | Evolved state, dt=0 | Evolved state, dt=0.01 and fixed F |
| --- | --- | --- | --- |
| Ferrite | SIGFPE | SIGFPE | finite stress and tangent |
| Austenite | SIGFPE | SIGFPE | finite stress and tangent |

The native probe is retained in the companion repository as `verification/zero_increment_probe.f90`. From that
repository, a reproduction sequence is:

```bash
conda run --no-capture-output -n py3.14 python build_bridge.py \
  --output build/zero_duration_audit/libmultiphase_trip.so
gfortran -O0 -g -ffpe-trap=invalid,zero,overflow \
  verification/zero_increment_probe.f90 build/zero_duration_audit/libmultiphase_trip.so \
  -o build/zero_duration_audit/zero_increment_probe
build/zero_duration_audit/zero_increment_probe bcc zero_initial
```

Invoke each case separately: phase is `bcc` or `fcc`; mode is `zero_initial`, `zero_evolved`, or `held_positive`.
The zero-duration modes are expected to raise SIGFPE; this is a diagnostic, not a supported integration call.

### Agreed resolution: observe accepted responses without constitutive evolution

Keep the strict positive-time ABI guard and the legacy constitutive kernel unchanged. No startup/restart tag or
zero-duration material update is needed. The supported cold start initializes history and zero stress. The FE solver
caches accepted Gauss-point Cauchy stress, saves it in restart schema 3, and reconstructs reporting/residuals from
geometry and this response cache. Restart restores full state and stress without reinitializing material points.
This cache is not an extra argument to the constitutive update. No material tangent or global stiffness is saved.

The next increment's first Newton trial uses the committed displacement as its initial guess but a positive elapsed
time. The material supplies fresh stress and tangent before the first correction for both Newton variants. Final
equilibrium checks reuse the accepted response; optional directional-tangent checks use the actual increment's
original committed history before commit, rather than reintegrating from its evolved endpoint at zero time.

The standalone material-point driver reports its initialized reference row without evolution. Tangent entries in
that row are `NaN` (unavailable); positive-time rows contain normal algorithmic tangents. Tests compare observed
responses for all three element formulations, reject zero-time calls through cold start/restart and tangent checks,
and verify stress reporting and resumed/cold solutions.

A bounded real-library check also passed for both ferrite and austenite: one Hex8-Fbar element with all boundary
nodes prescribed by affine simple shear, orientation entry zero, and two increments of duration `0.02` ending at
`F12=4e-5`. Restarting after the first increment into a separate results database reproduced the restart-time stress
exactly. Resumed and uninterrupted final displacement, complete state, and accepted stress were identical; force
balance residuals were below `2e-17`. This is a startup/restart integration check, not plastic/transformation
characterization or the pending periodic heterogeneous smoke case.
