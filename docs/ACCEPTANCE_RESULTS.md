# Acceptance results

Run on 2026-08-27 with Python 3.14, NumPy 2.5.2, SciPy 1.18.1, and Gmsh 4.15.2 using:

```bash
python -m verification.run_acceptance
python -m unittest discover -s tests -v
```

All five normative analyses landed on 20 mandatory `0.05` endpoints through `t = 1.0`. No analysis used an unintended cutback.

| Deck | Max Newton corrections | Min raw J | Force residual inf | Global tangent relative error |
|---|---:|---:|---:|---:|
| Case A Hex8 | 4 | 0.9873742694 | 2.91e-15 | 1.39e-9 |
| Case A Hex8-Fbar | 4 | 0.9667811152 | 6.78e-15 | 2.24e-9 |
| Case A Hex20 | 4 | 1.0000137569 | 1.18e-14 | 2.03e-9 |
| Case B Hex8 | 2 | 1.1000000000 | 6.65e-16 | 8.74e-10 |
| Case B Hex8-Fbar | 2 | 1.1000000000 | 5.01e-16 | 1.14e-9 |

Case B homogeneous checks at `t = 1.0`:

- maximum affine displacement error: `5.27e-16`;
- Hex8 versus Hex8-Fbar displacement error: `4.16e-16`;
- Cauchy-stress error: `7.06e-14`;
- constraint-reaction error: `9.66e-17`.

The unit suite contains 13 checks and passes in full. It includes centered material/element/augmented-global derivatives, an independent reference-versus-spatial F-bar tangent comparison, sparse nonsymmetric KKT solution, periodic spanning-tree construction, deliberate recoverable-failure cutback, and bitwise restart equivalence.
