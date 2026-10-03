# Multiphase TRIP integration plan

This document tracks the staged extension of the generic finite-strain solver. It is not a replacement for
`IMPLEMENTATION_SPEC.md`; requirements become normative there as their interfaces are implemented.

## Boundary check

The extension preserves the existing separation of concerns:

- **Preprocessing** owns domain-specific Voronoi parsing, voxel classification, phase/grain/orientation mapping, and
  construction of solver-ready mesh and immutable point-property data. Classification assigns a whole element to the
  cell containing its centroid. It does not call a constitutive routine.
- **Engine** receives prepared element blocks, phase material definitions, and immutable per-element point properties.
  It reconstructs deformation gradients from nodal kinematics, keeps complete committed/trial state at quadrature
  points, and remains unaware of geometry-file formats and output choices.
- **Postprocessing** reads only accepted HDF5 results. The deck explicitly selects reportable material-state fields;
  selection never changes integration or restart state. XDMF exports datasets generically and contains no FCC/BCC or
  crystal-plasticity-specific branches.

The two phases are two material models/definitions. Grain identity and orientation are point properties, not separate
material definitions. The external Fortran implementation remains opaque behind a wrapper returning the common
`P`, `dP/dF`, and packed-state response.

This implementation is the multiphase TRIP steel model: `multiphase_trip_ferrite` describes the ferritic matrix and
`multiphase_trip_austenite` includes austenite-to-martensite transformation. Integration temperature is fixed to the
compiled `Theta0` constant in `constants.for` (currently 300 K), with zero temperature increment. Changing it requires
a library rebuild; it is not a deck property.

## Domain preprocessing contract

- Voronoi domains occupy `[0,100]^3`; mesh resolution is an input and is not inferred from grain count.
- Merge repeated periodic cell images by `(phase, grain_id)` for classification and reporting.
- Build an explicit mapping from global Voronoi grain IDs to phase-local, zero-based orientation IDs. For the first
  dataset, phase 02 maps global IDs 1--8 to austenite orientations 0--7, and phase 01 maps global IDs 9--64 to ferrite
  orientations 0--55. Never encode the observed offset as a universal rule.
- Accept and report grains that contain no element centroid. Report element counts per grain and phase. Reject
  unclassified elements, invalid phase labels, and absent or out-of-range orientation mappings. Treat the source
  microstructure as periodic and do not impose an artificial discrete material-matching condition across opposite
  element layers; the Gmsh periodic node maps independently establish displacement periodicity.
- Keep neutral structured periodic mesh generation separate from centroid classification/deck decoration, while
  allowing one orchestration command to call both stages.

The available input families confirm the phase-local numbering convention without requiring filename uniformity:

| Dataset | Austenite global IDs / orientations | Ferrite global IDs / orientations |
| --- | --- | --- |
| 1A/7F | 1 / 1 | 2--8 / 1--7 |
| 8A/56F | 1--8 / 1--8 | 9--64 / 1--56 |
| 12A/88F | 1--12 / 1--12 | 13--100 / 1--88 |

The preprocessor will nevertheless construct and save the explicit mapping and expose zero-based orientation IDs to
the Python solver instead of assuming a fixed offset or phase count.

## Implementation stages

- [x] Add opt-in, material-scoped state-output selectors and generic HDF5/XDMF routing.
- [x] Update stateful example decks, tests, and restart/result-layout validation for the opt-in policy.
- [x] Add a generic immutable point-property source contract to the input deck and prepared model.
- [x] Propagate element properties through integration-point initialization, serial assembly, and process workers.
- [x] Store requested static cell properties such as `grain_id` and `orientation_id` and export them generically.
- [x] Implement and test the Voronoi/orientation classifier in the non-public companion repository.
- [x] Validate the classifier against the 8A/56F, 1A/7F, and 12A/88F datasets.
- [x] Generate a small periodic two-phase solver-ready mesh/deck and audit phase/grain/orientation counts.
- [x] Define and build the phase-specific multiphase TRIP wrapper ABI.
- [x] Register ferritic and austenitic models with `internal_variables` layouts of 93 and 151 entries respectively;
  their public component labels are zero-based.
- [x] Separate cold-start/restart observation from evolution; save accepted Gauss-point stress for reporting and
  evaluate all solver constitutive trials over positive-time increments.
- [ ] Add material-point stress/tangent/state verification before an FE solve.
- [ ] Run a bounded periodic FE smoke case, then prepare larger runs only after local verification passes.

Current interface review and blockers are recorded in `MULTIPHASE_TRIP_INTERFACE_REFINEMENT.md`. Runtime Euler angles
are proposed but not implemented. The legacy UMAT is demonstrably unsafe at zero duration; its strict positive-time
guard remains intact. FE observation now uses accepted stress, and the material-point driver's initial row reports
initialization without calling evolution. Tangent verification uses the actual positive-time increment.

## Output invariants

- No `[[output.material_state]]` entry means no time-dependent state field in `run.h5`.
- Restart files always contain the complete committed packed state, regardless of result selection.
- Restart schema 3 also retains accepted Gauss-point stress for reporting, but no tangents or independent `F` history.
- The full declared state layout remains in material metadata even when no values are selected.
- Rank-one state slices use declared component labels; ranges are inclusive and aliases are required for multi-component
  selections.
- The crystal-plasticity wrapper exposes the neutral field name `internal_variables`; names from the external ABI do
  not enter solver decks or result paths.
