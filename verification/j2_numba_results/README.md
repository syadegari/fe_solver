# J2 Numba feasibility evidence

This directory retains the small, machine-readable evidence that supported
adopting Numba as a required project dependency.  Large HDF5 solver outputs
remain ignored under `examples/results/j2_numba_feasibility/`.

The material benchmark executes the public material update, not only the
private numeric kernel.  It covers elastic and plastic points with and without
the algorithmic tangent, uses one BLAS thread, excludes the first call from
the warm samples, and checks a 400-increment sequential uniaxial path.  Both
backends execute the same numeric function; Numba compiles it in nopython mode
with `fastmath=False`.

The two bounded full-solver comparisons use the same small Hex8-Fbar prism,
two element processes, exact Newton, and residual backtracking:

- a startup solve from `t=0` to `t=0.02` after the compiled kernel cache had
  been created by the material benchmark;
- an evolved localized increment from the retained `t=0.9` restart to
  `t=0.9025`.

The generated JSON comparison tolerances are scale-aware through NumPy's
combined relative/absolute rule.  The retained absolute tolerance of
`2e-9` covers accumulated near-zero stress roundoff while remaining negligible
relative to the MPa-scale solution and solver tolerances.  Ordinary material,
element, and acceptance tolerances are unchanged.

## Observed result

On the two-core development laptop, median warm public-update speedups were:

| Synthetic material case | Speedup |
| --- | ---: |
| elastic stress/state | 3.87x |
| elastic with tangent | 6.63x |
| plastic stress/state | 7.30x |
| plastic with tangent | 10.85x |

Fixed-point stress was identical.  The largest fixed-point tangent difference
was `2.91e-11` (`5.29e-15` by the scale-aware relative measure).  After 400
committed uniaxial increments, the maximum stress, state, and tangent relative
differences were `2.79e-16`, `5.55e-17`, and `2.40e-16`, respectively.

The startup prism's recorded internal wall time fell from `9.788 s` to
`5.915 s` (1.65x).  In the localized restart comparison it fell from `8.081 s`
to `5.966 s` (1.35x), while summed element time fell from `5.499 s` to
`3.293 s` (1.67x).  Both comparisons followed identical increment, Newton,
and line-search paths.  Maximum displacement differences were below
`7.2e-15`; maximum Cauchy-stress differences were `4.91e-10 MPa` and
`1.05e-9 MPa`.

The evidence supports adopting the compiled J2 kernel for long J2 studies,
while retaining the interpreted implementation as the verification reference.
The speedup is dramatic at the material point; the remaining element
kinematics, tangent transformations, assembly transfer, and solver work limit
the complete-solver improvement.

Select it explicitly with `--j2-backend numba` on the ordinary solver or the
formulation-study runner.  Explicit selection keeps old commands on the
reference path and records the backend in `run_log.json`.
