# Circular-necking reference data

`circular_necking_reference.csv` transcribes the six literature columns in
Table 2 of the COMSOL 6.4 *Necking of an Elastoplastic Metal Bar* example:

<https://doc.comsol.com/6.4/doc/com.comsol.help.models.nsm.bar_necking/bar_necking.html>

COMSOL attributes the two columns to J. C. Simo and T. J. R. Hughes,
*Computational Inelasticity*, and T. Elguedj and T. J. R. Hughes,
*Isogeometric Analysis of Nearly Incompressible Large Strain Plasticity*, ICES
Report 11-35 (2011):

<https://www.ices.utexas.edu/media/reports/2011/1135.pdf>

These are numerical benchmark results, not physical-test measurements. The
published radii have only 0.1 mm precision. For the displacement plots they are
converted using

```text
u_r = current tabulated radius - 0.982 * 6.413 mm.
```

The separate displacement endpoint `u_r = -3.740 mm` at an imposed 7 mm is
the target published by ANSYS VM318, which attributes its reference to J. C.
Simo (1992):

<https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/ans_vm/Hlp_V_VM318.html>

ANSYS shows a full reference curve but supplies only this endpoint as tabular
acceptance data. Do not present the transcribed intermediate points or the
ANSYS endpoint as experimental observations.
