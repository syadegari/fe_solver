from __future__ import annotations

import numpy as np
from scipy import sparse

from .assembly import FEModel, assemble_external, assemble_internal, element_dofs
from .config import value_expression
from .constraints import ConstraintSystem
from .elements import evaluate_element
from .types import ElementRequest, ModelError


def _inf(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector, ord=np.inf)) if vector.size else 0.0


def _derivative_error(analytic: np.ndarray, finite_difference: np.ndarray) -> tuple[float, float]:
    absolute = _inf(analytic - finite_difference)
    scale = max(_inf(analytic), _inf(finite_difference), 1.0e-14)
    return absolute, absolute / scale


def verify_analysis(
    model: FEModel,
    constraints: ConstraintSystem,
    t: float,
    u: np.ndarray,
    lambdas: np.ndarray,
) -> dict[str, float]:
    options = model.deck.data.get("verification", {})
    summary: dict[str, float] = {}
    base = assemble_internal(model, u, u, t, t, True)
    f_ext = assemble_external(model, t)
    r_u = base.f_int - f_ext + constraints.C.T @ lambdas
    r_c = constraints.C @ u - constraints.rhs(t)
    summary["force_balance_inf"] = _inf(np.asarray(r_u))
    summary["constraint_residual_inf"] = _inf(np.asarray(r_c))
    controls = model.deck.data["nonlinear"]
    force_scale = max(_inf(base.f_int), _inf(f_ext), _inf(constraints.C.T @ lambdas))
    force_tol = float(controls["force_atol"]) + float(controls["force_rtol"]) * force_scale
    constraint_scale = max(_inf(constraints.C @ u), _inf(constraints.rhs(t)))
    constraint_tol = float(controls["constraint_atol"]) + float(controls["constraint_rtol"]) * constraint_scale
    if options.get("check_global_force_balance", False) and summary["force_balance_inf"] > force_tol:
        raise ModelError("accepted state fails global force balance verification")
    if summary["constraint_residual_inf"] > constraint_tol:
        raise ModelError("accepted state fails affine-constraint verification")

    minimum_J = min(value for block in base.gauss_output for element in block for value in element.J_raw)
    summary["minimum_J"] = float(minimum_J)
    if options.get("require_positive_J", False) and minimum_J <= 0.0:
        raise ModelError("accepted state fails positive-J verification")

    if options.get("affine_periodic", False):
        periodic = model.deck.data.get("constraints", {}).get("periodic_rve", [])
        if len(periodic) != 1:
            raise ModelError("affine-periodic verification requires one periodic_rve entry")
        entry = periodic[0]
        expressions = [[value_expression(entry["macro_F"][i][j]) for j in range(3)] for i in range(3)]
        Fbar = np.array(
            [[expressions[i][j].evaluate(t, model.deck.curves) for j in range(3)] for i in range(3)]
        )
        anchor = int(model.mesh.node_groups[str(entry["anchor_region"])][0])
        expected = (model.mesh.X - model.mesh.X[anchor]) @ (Fbar - np.eye(3)).T
        affine_error = _inf(u - expected.ravel())
        summary["affine_periodic_error_inf"] = affine_error
        if affine_error > float(options.get("affine_tolerance", 1.0e-9)):
            raise ModelError("periodic solution fails homogeneous affine-field verification")

    rng = np.random.default_rng(20240821)
    if options.get("check_element_directional_tangent", False):
        block = model.blocks[0]
        conn = block.connectivity[0]
        dofs = element_dofs(conn)
        direction = rng.normal(size=len(dofs))
        direction /= _inf(direction)
        eps = 1.0e-7 * max(1.0, _inf(u[dofs]))
        request = ElementRequest(
            model.mesh.X[conn], u[dofs], u[dofs], block.state_n[0], block.material,
            None, t, t, True, block.formulation,
        )
        analytic_response = evaluate_element(request)
        if not analytic_response.status.ok:
            raise ModelError(analytic_response.status.message)
        plus = evaluate_element(
            ElementRequest(
                model.mesh.X[conn], u[dofs], u[dofs] + eps * direction, block.state_n[0],
                block.material, None, t, t, False, block.formulation,
            )
        )
        minus = evaluate_element(
            ElementRequest(
                model.mesh.X[conn], u[dofs], u[dofs] - eps * direction, block.state_n[0],
                block.material, None, t, t, False, block.formulation,
            )
        )
        assert analytic_response.K is not None
        absolute, relative = _derivative_error(
            analytic_response.K @ direction, (plus.f_int - minus.f_int) / (2.0 * eps)
        )
        summary["element_tangent_absolute_error"] = absolute
        summary["element_tangent_relative_error"] = relative
        if relative > 2.0e-6 and absolute > 2.0e-8:
            raise ModelError("element tangent directional-derivative verification failed")

    if options.get("check_global_directional_tangent", False):
        assert base.K is not None
        direction = rng.normal(size=model.mesh.ndof + constraints.C.shape[0])
        direction /= _inf(direction)
        du = direction[:model.mesh.ndof]
        dlambda = direction[model.mesh.ndof:]
        eps = 1.0e-7 * max(1.0, _inf(u))

        def residual(displacement: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
            assembled = assemble_internal(model, u, displacement, t, t, False)
            return np.concatenate(
                [assembled.f_int - f_ext + constraints.C.T @ multipliers,
                 constraints.C @ displacement - constraints.rhs(t)]
            )

        finite_difference = (
            residual(u + eps * du, lambdas + eps * dlambda)
            - residual(u - eps * du, lambdas - eps * dlambda)
        ) / (2.0 * eps)
        kkt = sparse.bmat(
            [[base.K, constraints.C.T], [constraints.C, None]], format="csr"
        )
        absolute, relative = _derivative_error(np.asarray(kkt @ direction), finite_difference)
        summary["global_tangent_absolute_error"] = absolute
        summary["global_tangent_relative_error"] = relative
        if relative > 3.0e-6 and absolute > 3.0e-8:
            raise ModelError("global augmented tangent directional-derivative verification failed")
    return summary
