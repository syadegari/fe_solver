# External material libraries

A compiled integrator is connected through an ordinary Python material model.
The adapter translates arrays and status codes and returns the common
`MaterialResponse`. Elements and nonlinear algorithms continue to use the same
stress, tangent, state, and rollback contract described in [the material guide](MATERIALS.md).

## Supplied multiphase TRIP adapter

[`fe_solver/multiphase_trip.py`](../fe_solver/multiphase_trip.py) provides
`multiphase_trip_ferrite` and `multiphase_trip_austenite`. Both use the same external
shared library. The private constitutive source and library are not distributed
with this repository; archived results can be inspected without either.

```toml
[[materials]]
name = "matrix"
model = "multiphase_trip_ferrite"
[materials.properties]
library = "build/libmultiphase_trip.so"

[[point_property_sources]]
name = "microstructure"
file = "element_properties.h5"
format = "hdf5_element_properties"

[[element_assignments]]
region = "ferrite"
formulation = "hex8_fbar"
material = "matrix"
point_properties = "microstructure"
```

The library and property-file paths are relative to the TOML deck. The region is
a mesh Physical Group; `material` selects a material definition and
`point_properties` selects a property source. Those names need not match each other.
The adapter requires `phase_id` and a three-component `euler_angles` vector in
radians. Grain/orientation IDs are useful labels; the runtime update receives
the angles directly rather than looking them up inside the compiled library.

The general property loader matches HDF5 rows to Gmsh element tags. The material
adapter interprets the fields: phase 1 selects ferrite/BCC with 93 state values;
phase 2 selects austenite/FCC with 151. Neither mapping is a global FE convention.
The phase-specific wrappers validate each assignment before integration.

The adapter expects ABI version 5, with `cp_abi_version`, state-size queries, and
phase-specific `cp_bcc_initialize/update` and `cp_fcc_initialize/update` entry points.
The exact signatures and contiguous float64 layouts are specified in the adapter.
The shared library is loaded and cached separately in each process, with ABI,
state-size, and file-fingerprint checks. Calls receive endpoint `F` and times,
committed state, orientation, and `need_tangent`; they return `P`, `dP/dF` when
requested, and trial state. Temperature is the compiled 300 K constant for the
archived runs.

For another compiled model, write a separate adapter with its own properties,
state layout, ABI conversion, and failure mapping, then register it through the
same [material API](MATERIALS.md). An Abaqus UMAT signature or a spatial stress-rate
tangent cannot be substituted directly for the endpoint `P`/`dP/dF` contract.

## Inspecting and reproducing the TRIP examples

The [accepted results archive](../verification/multiphase_trip_results/README.md)
contains prepared decks, meshes, element properties, databases, logs, and plots.
Its README records library requirements, actual runtime overrides, and provenance
limitations. Raw Voronoi/orientation preprocessing and private-library compilation
remain documented in the companion repository's `PREPROCESSING.md`.

Internal variables are opaque to the FE solver and use Python's zero-based output
selection. For the supplied model, an austenitic material named `austenite` can request:

```toml
[[output.material_state]]
material = "austenite"
select = "internal_variables[91]"
name = "martensite_fraction"

[[output.material_state]]
material = "austenite"
select = "internal_variables[66]"
name = "beta_austenite"
```

These indices are model-specific, not general crystal-plasticity definitions.
The rate-dependent TRIP examples use a physical duration of 2000 seconds;
changing the duration can change the material response even with the same final `F`.
