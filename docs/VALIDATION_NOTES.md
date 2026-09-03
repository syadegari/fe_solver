# Validation notes

Checks performed on the final handoff:

- `FORMULATION.tex` compiles with pdfLaTeX from a clean directory in three passes.
- Final PDF length: 34 pages.
- No unresolved LaTeX references, overfull boxes, underfull boxes, or compile warnings were present in the final log scan.
- The complete PDF was rendered to page images and visually inspected, with focused checks on nomenclature, variational-guide, near-incompressibility, and algorithm sections.
- PDF preflight reports it as openable, unencrypted, text-based, and 34 pages.
- Final source scan found no supported-Hex27 references, no superscript `F^e/F^p` convention, no `t_{n+1}^{(i)}` pseudo-time notation, and no old `Delta v_g` quadrature-volume notation.
- `IMPLEMENTATION_SPEC.md` uses the same quadrature-weight convention `omega_g^x` as the formulation.
- Both Gmsh generator scripts pass `python -m py_compile`.
- All five TOML acceptance decks parse using Python `tomllib`.

Not executed in the preparation environment:

- Gmsh mesh generation itself, because the `gmsh` Python runtime was unavailable. The generators contain their own element/group/periodicity validation and should be run before FE implementation begins.
