# Finite-Strain FE Prototype - Codex Handoff

This package separates the mathematical reference from the coding contract.

## Read order for Codex

1. `AGENTS.md`
2. `docs/IMPLEMENTATION_SPEC.md`
3. Inspect `examples/`
4. Use `docs/FORMULATION.tex` only when mathematical meaning or a cited equation needs clarification. The PDF is the human-readable rendering.

The implementation specification is intentionally shorter and contains the final equations, data ownership rules, interfaces, algorithms, input schema, failure behavior, and acceptance tests. The formulation document contains the derivations and explanatory material.

If the two documents appear to conflict, stop and report the conflict rather than silently choosing a different formulation.

## v1 element scope

- full-integration Hex8
- centroidal F-bar Hex8
- full-integration Hex20 serendipity

The required first material is compressible neo-Hookean. A stateful finite-strain J2 model is a later plugin milestone. Crystal plasticity is outside the initial implementation milestone.

## Mesh layer

Gmsh remains the authoritative Python mesh/preprocessing layer, including Physical Groups and translational periodic node correspondence. The mechanical element kernels must remain independent of Gmsh.

## Important implementation invariants

- `F_n` and trial `F_np1` are reconstructed from nodal kinematics; `F` is not independently stored constitutive history.
- Every Newton trial integrates from the same committed material state at `n` to the current trial endpoint.
- Trial state is committed only after global convergence.
- Hex8-Fbar uses exactly the same material-point API as the standard elements; the element layer owns projection and chain-rule terms.
- General affine displacement constraints use the full sparse Lagrange-multiplier KKT system. Do not eliminate dependent displacement DOFs.
- Mandatory load-curve knots, output times, restart times, and `t_end` are solved equilibrium endpoints.

## Environment-validation note

The formulation LaTeX was compiled from a clean directory through three passes and rendered for visual inspection. The example TOML files parse with Python `tomllib`, and both Gmsh generator scripts pass `py_compile`.

The `gmsh` Python runtime was not available in the preparation environment, so the mesh generators could not be executed here. The first implementation step should therefore run and validate both generators in the Codex environment before depending on their output.
