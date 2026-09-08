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
