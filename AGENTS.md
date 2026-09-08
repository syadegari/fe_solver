# Project instructions

Implement the finite-strain FE prototype defined by `docs/IMPLEMENTATION_SPEC.md`.

Mathematical reference: `docs/FORMULATION.tex` (PDF copy is for human reading). The implementation specification is the coding contract; the formulation reference explains and justifies the equations. If they appear to conflict, stop at the conflict and report it rather than silently choosing a different formulation.

## Required v1 scope

- standard full-integration Hex8
- centroidal F-bar Hex8
- standard full-integration Hex20
- compressible neo-Hookean reference material
- common material-point API based on `F_n`, trial `F_np1`, state, `P`, and algorithmic `dP/dF`
- sparse global assembly
- general affine displacement constraints through the Lagrange-multiplier KKT system; do not eliminate dependent displacement DOFs
- exact and modified Newton
- committed/trial material state with rollback and cutback
- Gmsh Physical Groups and Gmsh periodic node maps
- pseudo-time curves, mandatory output/restart events, restart
- supplied acceptance cases


## Non-negotiable invariants

- `F` is reconstructed from reference coordinates and nodal kinematics. It is not independently advanced constitutive history.
- Every Newton trial integrates the material from the same committed state at `n` to the current trial endpoint at `n+1`.
- Trial material state is not committed before global convergence.
- Hex8-Fbar uses the same material API as standard elements. The element wrapper owns the F-bar projection and its tangent chain rule.
- Do not assume tangent or KKT symmetry.
- Use sparse global matrices. Dense global matrices are allowed only in deliberately small tests.
- The required constraint path is the full KKT system, not displacement elimination.
- Do not weaken verification or acceptance criteria merely to make tests pass.

## Development behavior

### Python environment

- Run project Python commands, tests, examples, and Gmsh generators in the Conda environment `py3.14`.
- Prefer `conda run -n py3.14 python ...` (or the interpreter at `/home/srn/miniconda3/envs/py3.14/bin/python`) because separate tool calls do not retain `conda activate` state.
- Before reporting a Python dependency as unavailable, check it inside `py3.14`.
- The user authorizes installing missing project packages into `py3.14` with `conda run -n py3.14 python -m pip install ...`; platform sandbox approval may still be required for the write.

Read `docs/IMPLEMENTATION_SPEC.md` before broad implementation. Inspect the supplied example mesh generators and TOML decks. Implement verification tests alongside each numerical component, especially finite-difference material and element tangents.

Small exploratory scripts are encouraged when they establish a convention or test a formula. If the specification is incomplete or appears mathematically inconsistent, report the issue before building dependent code.

Get the neo-Hookean acceptance problems working before adding a stateful J2 model. Do not begin crystal-plasticity integration in the initial prototype.
