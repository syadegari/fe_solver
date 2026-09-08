# Formulation normalization and audit notes

This formulation is the normalized successor to the earlier combined teaching/implementation manuscript.

2026-09-08 J2 solver-diagnostics follow-up:

- added optional residual-based Newton backtracking to the generic KKT solution method, including the effect of damping on affine-constraint residuals and the requirement that every candidate reuse committed material history;
- retained the distinction between exact and modified Newton: damping changes the search length, while exact Newton still uses the tangent at each accepted iterate;
- kept material-point characterization, diagnostic scripts, mesh sizes, and line-search input controls in `IMPLEMENTATION_SPEC.md` rather than turning them into constitutive theory.

2026-09-07 finite-strain J2 extension:

- added the complete multiplicative J2/Voce constitutive example as a self-contained appendix, preserving the material-agnostic focus of the main formulation;
- documented the radial return, determinant-one plastic-metric reconstruction, and exact directional construction of `dP/dF` used by the implementation;
- added the corresponding coding contract and circular-bar necking benchmark only to `IMPLEMENTATION_SPEC.md`.

2026-09-07 output and frame-benchmark refinement:

- derived the positive transverse stretch that makes the neo-Hookean bar's lateral surfaces traction-free;
- clarified end-face-only motion and the expected axial Cauchy-stress transfer under a 90-degree rotation;
- distinguished unscaled six-component reporting from engineering-shear assembly, including double-contraction weights;
- kept HDF5/XDMF schema and boundary-path implementation rules in `IMPLEMENTATION_SPEC.md`.

Key changes in the final normalization:

- separated the mathematical formulation from the coding contract;
- added a notation-conventions table and expanded nomenclature near the front;
- standardized the spatial velocity gradient and its split as `L = d + w`;
- standardized fourth-order tangents as blackboard-bold `A` and lower-case blackboard-bold `c`, while finite-dimensional vectors/matrices use upright bold notation;
- reserved upright superscript `T` for the Truesdell representation and `top` for transpose;
- standardized elastic/plastic deformation-gradient factors as `F_e` and `F_p` with subscripts;
- retained `int` and `ext` as descriptive superscripts on internal/external discrete quantities;
- explicitly defined the algorithmically returned endpoint stress `P-hat` and the derivative `D_{F_np1} P-hat`;
- removed Hex27, leaving Hex8, Hex8-Fbar, and Hex20;
- replaced programming-array notation inside the mathematical formulation with node-block/tensor notation;
- added a reading guide before the variational formulation and a detailed appendix on variations, dependency bookkeeping, the product rule, test fields versus Newton directions, spatial-gradient variation, and constitutively condensed dependencies;
- made the current Newton-trial configuration explicit wherever spatial gradients/Jacobians are evaluated;
- made deformation gradients derived kinematic quantities everywhere, not stored Gauss-point history;
- corrected KKT solvability language: for full-row-rank `C`, local constrained solvability is governed by nonsingularity of `Z^T K Z`, not merely by KKT indefiniteness;
- corrected the homogeneous Hex8/Hex8-Fbar statement: residual/stress/state agree, and affine tangent action agrees, but the complete tangents need not be identical for arbitrary non-affine perturbations;
- retained the full consistent centroidal F-bar chain rule and the common material interface;
- removed implementation-specific file/package details from the main formulation where they interrupted the research-style progression.
