# Validation notes

Checks performed on the current handoff:

- `FORMULATION.tex` compiles with pdfLaTeX from a clean directory in three passes.
- Final PDF length: 38 pages after adding the self-contained J2 appendix.
- No unresolved LaTeX references, overfull boxes, underfull boxes, or compile warnings were present in the final log scan.
- The complete PDF was rendered to page images and visually inspected, with focused checks on nomenclature, variational-guide, near-incompressibility, and algorithm sections.
- PDF preflight reports it as openable, unencrypted, text-based, and 38 pages.
- Final source scan found no supported-Hex27 references, no superscript `F^e/F^p` convention, no `t_{n+1}^{(i)}` pseudo-time notation, and no old `Delta v_g` quadrature-volume notation.
- `IMPLEMENTATION_SPEC.md` uses the same quadrature-weight convention `omega_g^x` as the formulation.
- All five Gmsh generator scripts pass `python -m py_compile`.
- All ten TOML decks parse using Python `tomllib`.

J2 checks performed with the available NumPy runtime:

- all nine `dP/dF` columns agree with centered finite differences on representative elastic and plastic branches; relative Frobenius errors were `3.93e-10` and `5.76e-10`;
- yielded standard Hex8 and Hex8-Fbar tangent actions agree with centered residual differences; relative infinity-norm errors were `1.86e-8` and `4.75e-8`;
- the returned plastic metric remained symmetric positive-definite and unit-determinant, repeated evaluation at the accepted endpoint did not advance state, and a superposed rotation transformed first Piola stress covariantly.
- the complete 34-test suite passes in the `py3.14` Conda environment, including prescribed-F material paths and rejection of an invalid full Newton candidate without pseudo-time cutback;
- the eight-deck default acceptance suite passes with zero cutbacks; its material, element, and global tangent checks remain below `4.76e-9` relative error, and the frame-objectivity Cauchy-stress rotation error is `4.36e-14`;
- Gmsh 4.15.2 generated the three-block MSH 4.1 benchmark with 1,300 nodes and 960 Hex8 volume elements, all required Physical Groups, positive reference Jacobians, and minimum scaled Jacobian `0.668`;
- the small square-prism generator produced 225 nodes and 96 Hex8 volume elements with the same diagnostic Physical Groups and minimum scaled Jacobian `0.998989`;
- a one-increment end-to-end benchmark smoke test converged to `t=0.02` with `max|u|=0.14` and normalized force-balance residual `1.03e-9`.

Focused investigation of the post-yield slowdown:

- 400-step prescribed-F histories were generated for 10% isochoric logarithmic uniaxial extension and `F12=0.1` simple shear, with 200-step endpoint comparisons. Relative final stress differences were `1.76e-5` and `1.42e-4`; relative final state differences were `6.83e-6` and `5.12e-7`.
- at the circular-bar state advanced from restart `t=0.4` to accepted output `t=0.405`, centered directional checks over three representative evolved elements gave maximum relative errors `3.76e-9` for the J2 material tangent and `6.06e-9` for the complete Hex8-Fbar element tangent;
- a 96-element tapered square-prism diagnostic reproduced the undamped failure on `t=0.43 -> 0.44`: the force residual rose from `2.32e3` to `6.61e3`, the reduced-tangent condition estimate reached `5.38e5`, and the following full step produced an invalid trial centroid Jacobian;
- residual backtracking converged that same reconstructed diagnostic increment in 13 iterations. A sustained run reached `t=0.5` in 26 accepted increments with no cutbacks and normalized force-balance residual `5.44e-9`. The original `dt=0.02` was retained except where mandatory events split it, so no time-growth policy change was justified.

Not executed in the preparation environment:

- the full circular-bar necking history with Newton backtracking. The benchmark remains a long acceptance run in the current pure-Python implementation.

## Periodic elastic-core/J2-matrix feature validation (2026-09-14)

The production-resolution isochoric-uniaxial and simple-shear cases were run on a `16 x 16 x 16` periodic cube with
a centered `8 x 8 x 8` core.  The conforming mesh has 4,913 nodes and 4,096 elements: 3,584 J2/Voce Hex8-Fbar
matrix elements and 512 neo-Hookean standard Hex8 core elements.  Both databases contain 62 complete accepted
states from `t=0` through `t=1`; each solve used 61 increments, no cutbacks, and at most seven Newton iterations.

| quantity | isochoric uniaxial | simple shear |
| --- | ---: | ---: |
| final imposed macro component | `F11 = 1.2` | `F12 = 0.2` |
| force-balance infinity norm | `4.905e-12` | `5.296e-12` |
| constraint-residual infinity norm | `2.776e-17` | `2.776e-17` |
| minimum material-point `J` | `0.651147` | `0.945420` |
| independent maximum `|<F_raw>_0-Fbar|_inf` | `5.329e-14` | `1.843e-14` |
| final macroscopic nominal component [MPa] | `P11 = 483.996` | `P12 = 414.261` |
| final matrix-average Cauchy component [MPa] | `sigma11 = 427.022` | `sigma12 = 388.281` |
| final core-average Cauchy component [MPa] | `sigma11 = 1650.600` | `sigma12 = 596.118` |
| elapsed wall time, 16 element workers | `2:54:47` | `2:48:16` |

All saved nodal, stress, strain, and material-state arrays are finite.  The HDF5 and XDMF outputs retain separate
matrix/core branches, and only the matrix contains `equivalent_plastic_strain` and `plastic_metric_inverse`.  The
phase-average histories show the expected higher stress in the elastic core after matrix yielding.  ParaView's XDMF
Reader T was visually checked on the `4/2` smoke result: phase selections persist throughout playback using the stable
phase-before-time hierarchy.  VTK does not propagate temporal collection names, so the branches appear as deterministic
`Block0` (matrix) and `Block1` (core).

The remote source snapshot was prepared with `git archive` from revision `8e52a01`.  Because an archive has no `.git`
directory, the result databases record `git_commit="unknown"`; independent recomputation confirms that both stored
model-identity hashes exactly match the current 16/8 meshes and decks.  Local postprocessing used revision `a955041`.
The detailed local artifacts are intentionally ignored by Git and reside under the corresponding
`examples/results/periodic_core_j2_matrix_*` directories: `run.h5`, `run_log.json`, `terminal.log`,
`periodic_composite_history.json`, its PNG plot, and the XDMF entry point/sidecar.

The 48-test unit suite and the default eight-deck acceptance suite pass after these changes.  All eight acceptance
decks complete without cutbacks.  Timing telemetry from the production runs attributes approximately 92.5 percent of
wall time to sparse KKT factorization and approximately 6.4 percent to all element phases, identifying the global
linear solver rather than element processing as the dominant performance limit for these cases.
