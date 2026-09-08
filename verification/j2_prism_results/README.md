# Small J2 prism: Hex8-Fbar versus Hex8

`hex8_vs_hex8_fbar.png` is the first completed paired comparison on the common 96-element square-prism mesh. Both runs use the same geometry, material, constraints, load curve, nonlinear controls, and two-process element backend. Only the element formulation differs. The figure was generated with:

```bash
python -m verification.check_j2_prism \
  examples/results/j2_necking_prism_small_hex8_fbar/run.h5 \
  --compare examples/results/j2_necking_prism_small_hex8/run.h5 \
  --summary-only --plot verification/j2_prism_results/hex8_vs_hex8_fbar.png
```

The source HDF5 databases are run artifacts and are not committed. Their SHA-256 identities for this figure are:

- Hex8-Fbar: `88b80b54b856e3c64c464bcc6e548f7c7ed3a1520c1a90d664c6b9493dd3bcc3`
- Hex8: `043547b4891b50b4235eea6e087b301df0ddbd79d93508fba7f0b81063bb27d6`

## Selected results

| Observable | Hex8-Fbar | Hex8 |
| --- | ---: | ---: |
| Peak end reaction [N] | 19,332.68 at `t=0.40` | 19,459.58 at `t=0.48` |
| End reaction at `t=1` [N] | 8,361.67 | 16,617.13 |
| Middle mean half-width at `t=1` [mm] | 2.5480 | 4.1866 |
| Maximum equivalent plastic strain at `t=1` | 1.4840 | 0.6162 |
| Raw Gauss-point `J` range at `t=1` | 0.7336--1.2886 | 0.9863--1.0167 |
| Material-seen `J` range at `t=1` | 0.9981--1.0040 | 0.9863--1.0167 |
| Maximum section mean-stress spread at `t=1` [MPa] | 570.93 | 219.63 |
| Accepted increments | 159 | 66 (31 + 35 restart stages) |
| Cutbacks | 3 | 3 (2 + 1 restart stages) |
| Maximum converged Newton iterations | 22 | 21 |
| Line-search trials / backtracks | 2,257 / 1,056 | 1,577 / 953 |
| Solver-recorded elapsed wall time [s] | 2,543.21 | 1,824.33 (255.34 + 1,569.00) |

At final elongation, standard Hex8 carries 98.7% more reaction, retains a 64.3% larger middle half-width, and develops 58.5% less maximum equivalent plastic strain. This is qualitative evidence of volumetric locking suppressing neck localization on this coarse mesh. It is not a converged reference comparison; quantitative conclusions require a paired mesh-refinement study.
