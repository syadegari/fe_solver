# Adding a material model

Elements call a common material-point interface. A model receives the committed
and trial deformation gradients, committed internal variables, properties, and
increment times. It returns first Piola–Kirchhoff stress, the algorithmic tangent,
a trial state, and an evaluation status. The element owns the spatial tangent
transformation and, for Hex8-Fbar, the projection and its chain rule.

## The interface

| Input/output | Meaning |
| --- | --- |
| `request.F_n`, `request.F_np1` | Deformation gradients at the committed start and current trial endpoint; each has shape `(3, 3)`. |
| `request.state_n` | Named view of committed history, with a flat `.values` array. Treat it as read-only. |
| `request.properties` | Constants shared by a material definition. |
| `request.point_properties` | Optional immutable element/point properties, such as orientation. |
| `request.t_n`, `request.t_np1` | Increment endpoint times; their difference matters for rate-dependent models. |
| `request.need_tangent` | Whether this call requires the tangent. Residual-only evaluations may return `None`. |
| `response.P` | First Piola–Kirchhoff stress, shape `(3, 3)`. |
| `response.A_alg` | Shape `(3, 3, 3, 3)`, with `A_alg[i,I,j,J] = dP[i,I]/dF[j,J]` for the discrete update. |
| `response.state_trial` | Trial history in the model's declared `StateLayout`. |
| `response.status` | Success, recoverable trial failure, or fatal model/input failure. |

For Newton iteration and every line-search candidate, the solver evaluates the
same interval from the same committed history. Do not advance an integrator from
the previous Newton candidate, mutate `state_n`, or independently advance `F` as
history. Global convergence is the point at which the solver commits trial state.
Cold-start/restart reporting uses initialized/restored data without an update call.

## Minimal worked example

The following stateless compressible neo-Hookean model deliberately duplicates
the supplied reference law to illustrate the complete registration path. Save it
as `custom_materials.py` in the repository root. No solver or element changes are
needed.

```python
import numpy as np

from fe_solver.materials import register_material_model
from fe_solver.types import (
    EvaluationStatus, FailureKind, MaterialModel, MaterialResponse,
    ModelError, StateLayout,
)


def validate_custom_neo_hook(properties):
    if set(properties) != {"mu", "kappa"}:
        raise ModelError("custom_neo_hook requires exactly mu and kappa")
    values = {name: float(value) for name, value in properties.items()}
    if any(not np.isfinite(value) or value <= 0 for value in values.values()):
        raise ModelError("mu and kappa must be finite and positive")
    return values


def update_custom_neo_hook(request):
    F = np.asarray(request.F_np1, dtype=float)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)):
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n,
            EvaluationStatus(FailureKind.RECOVERABLE, "invalid trial F"),
        )
    J = float(np.linalg.det(F))
    if J <= 0 or not np.isfinite(J):
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n,
            EvaluationStatus(FailureKind.RECOVERABLE, "nonpositive trial J"),
        )
    try:
        H = np.linalg.inv(F).T
    except np.linalg.LinAlgError:
        return MaterialResponse(
            np.zeros((3, 3)), None, request.state_n,
            EvaluationStatus(FailureKind.RECOVERABLE, "singular trial F"),
        )
    mu, kappa = request.properties["mu"], request.properties["kappa"]
    logJ = np.log(J)
    P = mu * (F - H) + kappa * logJ * H
    A = None
    if request.need_tangent:
        eye = np.eye(3)
        A = (
            mu * np.einsum("ij,IJ->iIjJ", eye, eye)
            + kappa * np.einsum("iI,jJ->iIjJ", H, H)
            + (mu - kappa * logJ) * np.einsum("iJ,jI->iIjJ", H, H)
        )
    return MaterialResponse(P, A, request.state_n, EvaluationStatus())


register_material_model(MaterialModel(
    root="custom_neo_hook",
    update=update_custom_neo_hook,
    initialize=None,
    validate_properties=validate_custom_neo_hook,
    state_layout=StateLayout(),
))
```

The naming contract is `update_<root>` and, when history exists, `init_<root>`.
A stateless model declares an empty layout and no initializer; initialization is
then skipped. Use named, module-level functions so worker processes can import them.

Copy `examples/case_a_hex8.toml` to `examples/custom_neo_hook.toml`. In that copy,
change the existing material's `model` to `custom_neo_hook` and set a distinct
output directory. Keep its material name and element assignment unchanged:

```toml
[[materials]]
name = "matrix"
model = "custom_neo_hook"
[materials.properties]
mu = 1.0
kappa = 20.0

[[element_assignments]]
region = "solid"
formulation = "hex8"
material = "matrix"
```

Registration is performed by Python import; a TOML model name does not load a
Python module automatically. Save this launcher as `run_custom.py` beside
`custom_materials.py`:

```python
import sys
import custom_materials  # registers the model before preprocessing
from fe_solver.solver import run_analysis

if __name__ == "__main__":
    run_analysis(sys.argv[1])
```

Run it from the repository root in the project environment:

```bash
python run_custom.py examples/custom_neo_hook.toml
```

## History-dependent models

Declare named fields with `StateLayout` and `StateField`, and supply an initializer
returning `MaterialInitResponse(state0, status)`. For example, a scalar history
field uses `StateLayout((StateField("accumulated_slip"),))`; its initializer must
return a flat array of length one. Layouts can also contain vectors with declared
component labels. See the [J2 implementation](../fe_solver/materials.py) for a
complete stateful model.

Inside the update, copy `request.state_n.values` before changing history and
return a new view using the same layout. All constitutive quantities needed for
future integration, including prior stress if the model needs it, belong in that
history. The solver's reporting stress cache is not an integrator input.

State output is selected independently of restart storage. No internal variables
are written to the results database unless requested. For example, the supplied
J2 model can expose a named scalar with:

```toml
[[output.material_state]]
material = "steel"
select = "equivalent_plastic_strain"
name = "plastic_strain"
```

Vector fields support zero-based component and slice selections such as
`internal_variables[57]` or `internal_variables[33:58]`. Non-singleton slices need
an output name. Restart checkpoints retain the complete integration state,
regardless of which fields are selected for visualization.

## Checking a new implementation

Start with a material-point path before running a mesh. The generic
[`run_material_path`](../fe_solver/material_point.py) driver accepts a target `F`
or a prescribed path and a number of steps, initializes once, and records stress,
strain, state, and optional tangents. Check elastic limits, expected loading
responses, state admissibility, and increment-size sensitivity.

Check `dP/dF` against directional finite differences of the discrete update using
fresh copies of the same committed history for both perturbations. Near active-set
switches or substepping boundaries, interpret the derivative against the model's
documented algorithm rather than assuming a smooth map. Then check element tangents,
rotation behavior, failed-trial rollback, restart, and a bounded FE example.
Existing [numerical tests](../tests/test_numerics.py) demonstrate these conventions.

Return a recoverable status for a trial configuration or local integration failure
that a smaller step may resolve; return a fatal status for invalid model data or
an incompatible interface. Error status must not commit partial history.

Compiled constitutive routines use the same FE contract through a Python adapter;
see [external material libraries](EXTERNAL_MATERIALS.md).
