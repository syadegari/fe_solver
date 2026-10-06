# 8A56F_8x8x8_simple_shear

Saved interval: 0 to 2000 seconds;
requested end: 2000 seconds. There are 528 saved states.
Source run commit: `5b8582c2275707563e484ac34a2e95e0a84f1830`.
Initial phase volume fractions: ferrite: 88.0859%, austenite: 11.9141%.

Plots are saved in SVG format only.

`stress_vs_strain.svg`: Cauchy stress component 12, averaged
over each phase and the entire RVE using current element volumes reconstructed
from the saved nodal displacement. Native TRIP stress is GPa; report stress is MPa.
The horizontal axis is hbar11 = ln(lambdabar) for isochoric tension, and 2 hbar12
for simple shear. Here hbar = (1/2) log(Fbar Fbar^T), so
hbar12 = asinh(gamma/2) / sqrt(1 + gamma^2/4). The factor of two is only the shear
plot-axis convention; the original hbar12 component is also retained in the data.
It is not log(1+gamma), nor an average of local logarithmic strains.
These scalar plots are not presented as work-conjugate stress/strain pairs.
The original engineering strain and equivalent macro Hencky strain remain in
the saved numerical data.

To evaluate the matrix logarithm, diagonalize Bbar = Fbar Fbar^T as
Bbar = Q diag(b1,b2,b3) Q^T, then form
h = Q diag(0.5 ln(b1), 0.5 ln(b2), 0.5 ln(b3)) Q^T.
The bi are squared principal stretches, so this takes the logarithm of each
principal stretch and transforms back to the fixed spatial axes. Every subplot
uses the same loading-axis strain (2 hbar12 for shear or hbar11 for tension), not a different strain for
each stress component. The input is prescribed macro F, not a material elastic
gradient or the element's projected F-bar gradient.

`stress_components_vs_strain.svg`: six subplots in order
[11,22,33,12,23,13], each showing ferrite, austenite, and whole-RVE means.

`von_mises_vs_strain.svg`: left, von Mises of each averaged stress tensor;
right, current-volume mean of the local cell von Mises stresses. In both cases
sigma_vm = sqrt(3 J2), with J2 = (s:s)/2 and s = sigma - tr(sigma) I/3.
Tensor contraction counts off-diagonal shear terms twice. Averaging and taking
this invariant do not commute, so the two panels measure different things.
Von Mises is a descriptive stress invariant here, not the TRIP model's yield
criterion. Its values are saved in MPa; J2 itself would have stress-squared units.

`martensite_vs_strain.svg`: the mean saved `martensite_fraction`
within the initially austenitic region, and its contribution per initial RVE volume.
Both use reference-volume weights; the latter treats initial ferrite as zero.
These summarize the stored SigmaXi variable, not a reconstruction of the exact
current martensite volume fraction accounting for transformation dilatation.
Numeric fractions are 0--1; plots show percentages.

`beta_vs_strain.svg`: reference-volume phase means of the saved beta
outputs (field aliases `beta` or `beta_*`). Beta is the constitutive model's
internal parameter, not an independently defined equivalent plastic strain.

## Periodic kinematic average

The maximum componentwise error of <F_raw>_V0 - Fbar over all saved states is
8.104628e-15. Raw F is reconstructed at
the eight quadrature points from reference coordinates and nodal displacement;
integration uses reference-volume weights, not projected material F-bar gradients.
All nine average and prescribed components, per-state errors, and the 3x3 table
of maximum component errors are retained in `history.json`/`history.npz`.

`history.json`: labeled arrays and averaging/unit metadata.
`history.npz`: the same numeric columns for NumPy (`numpy.load`). Saved scalar
state averages, including the selected phase beta outputs, are also included.
All averages use the saved centroid-recovered fields and are postprocessing
approximations; no material library, original properties file, or mesh file is needed.
No source dataset is modified.

Reproduce from the repository root:

```bash
conda run --no-capture-output -n py3.14 python -m verification.summarize_multiphase_trip \
  PATH_TO_RUN_H5 --output PATH_TO_REPORT_DIRECTORY
```

For renamed transformation output, also supply `--martensite-field FIELD_NAME`.
The same command supports both simple-shear and isochoric-uniaxial databases.
