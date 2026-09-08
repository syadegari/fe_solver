from __future__ import annotations

import numpy as np

from .materials import evaluate_material_point
from .quadrature import HEX20_POINTS, HEX20_WEIGHTS, HEX8_POINTS, HEX8_WEIGHTS
from .shape import hex20_shape, hex8_shape
from .tangents import truesdell_voigt
from .types import (
    ElementRequest,
    ElementResponse,
    EvaluationStatus,
    FailureKind,
    GaussOutput,
    MaterialRequest,
)


_GEOM_TOL = 1.0e-12


def _failed(request: ElementRequest, message: str, kind: FailureKind) -> ElementResponse:
    ndof = request.u_e_trial.size
    return ElementResponse(
        np.zeros(ndof), None, request.state_e_n.copy(), EvaluationStatus(kind, message), None
    )


def _kinematics(
    X: np.ndarray, x: np.ndarray, dN: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    J0 = X.T @ dN
    Jx = x.T @ dN
    detJ0 = float(np.linalg.det(J0))
    detJx = float(np.linalg.det(Jx))
    invJ0 = np.linalg.inv(J0)
    grad_X = dN @ invJ0
    F = Jx @ invJ0
    return J0, Jx, detJ0, detJx, grad_X, F


def _quadrature(formulation: str):
    if formulation in ("hex8", "hex8_fbar"):
        return hex8_shape, HEX8_POINTS, HEX8_WEIGHTS
    if formulation == "hex20":
        return hex20_shape, HEX20_POINTS, HEX20_WEIGHTS
    raise ValueError(f"unsupported formulation {formulation!r}")


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def _B_matrix(grad_x: np.ndarray) -> np.ndarray:
    B = np.zeros((6, 3 * len(grad_x)))
    for a, (g0, g1, g2) in enumerate(grad_x):
        B[:, 3 * a:3 * a + 3] = (
            (g0, 0.0, 0.0),
            (0.0, g1, 0.0),
            (0.0, 0.0, g2),
            (g1, g0, 0.0),
            (0.0, g2, g1),
            (g2, 0.0, g0),
        )
    return B


def _stress_voigt(sigma: np.ndarray) -> np.ndarray:
    return np.asarray([sigma[i, j] for i, j in _VOIGT_PAIRS])


def evaluate_standard_element(request: ElementRequest) -> ElementResponse:
    shape, points, weights = _quadrature(request.formulation)
    X = np.asarray(request.X_e, dtype=float)
    nnode = len(X)
    if request.u_e_n.shape != (3 * nnode,) or request.u_e_trial.shape != (3 * nnode,):
        raise ValueError("element displacement vector has the wrong shape")
    x_n = X + request.u_e_n.reshape(nnode, 3)
    x = X + request.u_e_trial.reshape(nnode, 3)
    f = np.zeros(3 * nnode)
    K = np.zeros((3 * nnode, 3 * nnode)) if request.need_tangent else None
    trial_state = np.empty_like(request.state_e_n)
    output = GaussOutput()

    for g, (xi, weight) in enumerate(zip(points, weights)):
        N, dN = shape(xi)
        del N
        try:
            _, _, detJ0, detJx_n, grad_X, F_n = _kinematics(X, x_n, dN)
            _, _, _, detJx, _, F = _kinematics(X, x, dN)
        except np.linalg.LinAlgError:
            return _failed(request, f"singular element mapping at Gauss point {g}", FailureKind.FATAL)
        if not np.isfinite(detJ0) or detJ0 <= _GEOM_TOL:
            return _failed(request, f"nonpositive reference Jacobian at Gauss point {g}", FailureKind.FATAL)
        if not np.isfinite(detJx_n) or detJx_n <= _GEOM_TOL:
            return _failed(request, f"invalid committed Jacobian at Gauss point {g}", FailureKind.FATAL)
        if not np.isfinite(detJx) or detJx <= _GEOM_TOL:
            return _failed(request, f"invalid trial Jacobian at Gauss point {g}", FailureKind.RECOVERABLE)
        J = float(np.linalg.det(F))
        response = evaluate_material_point(
            request.material,
            MaterialRequest(
                F_n, F, request.material.state_layout.view(request.state_e_n[g]), request.material.properties,
                request.point_properties, request.t_n, request.t_np1, request.need_tangent,
            ),
        )
        if not response.status.ok:
            return _failed(request, response.status.message, response.status.kind)
        trial_state[g] = response.state_trial.values
        grad_x = grad_X @ np.linalg.inv(F)
        B = _B_matrix(grad_x)
        sigma = response.P @ F.T / J
        dv = detJx * float(weight)
        f += B.T @ _stress_voigt(sigma) * dv
        if K is not None:
            assert response.A_alg is not None
            D = truesdell_voigt(response.A_alg, F, J, sigma)
            K += B.T @ D @ B * dv
            K += np.kron(grad_x @ sigma @ grad_x.T * dv, np.eye(3))
        output.F_raw.append(F.copy())
        output.J_raw.append(J)
        output.F_material.append(F.copy())
        output.J_material.append(J)
        output.P_material.append(response.P.copy())
        output.P_effective.append(response.P.copy())
        output.cauchy_stress.append(sigma)
    return ElementResponse(f, K, trial_state, EvaluationStatus(), output)


def evaluate_fbar_element(request: ElementRequest) -> ElementResponse:
    if request.formulation != "hex8_fbar" or request.X_e.shape != (8, 3):
        raise ValueError("F-bar is defined only for Hex8")
    X = np.asarray(request.X_e, dtype=float)
    x_n = X + request.u_e_n.reshape(8, 3)
    x = X + request.u_e_trial.reshape(8, 3)
    _, dN_c = hex8_shape(np.zeros(3))
    try:
        _, _, detJ0_c, detJx_n_c, grad_X_c, Fc_n = _kinematics(X, x_n, dN_c)
        _, _, _, detJx_c, _, Fc = _kinematics(X, x, dN_c)
    except np.linalg.LinAlgError:
        return _failed(request, "singular centroid mapping", FailureKind.FATAL)
    if detJ0_c <= _GEOM_TOL:
        return _failed(request, "nonpositive reference centroid Jacobian", FailureKind.FATAL)
    if detJx_n_c <= _GEOM_TOL:
        return _failed(request, "invalid committed centroid Jacobian", FailureKind.FATAL)
    if detJx_c <= _GEOM_TOL:
        return _failed(request, "invalid trial centroid Jacobian", FailureKind.RECOVERABLE)
    Jc_n = float(np.linalg.det(Fc_n))
    Jc = float(np.linalg.det(Fc))
    if Jc_n <= 0.0 or Jc <= 0.0:
        kind = FailureKind.FATAL if Jc_n <= 0.0 else FailureKind.RECOVERABLE
        return _failed(request, "invalid F-bar centroid determinant", kind)

    f = np.zeros(24)
    K = np.zeros((24, 24)) if request.need_tangent else None
    trial_state = np.empty_like(request.state_e_n)
    output = GaussOutput()
    for g, (xi, weight) in enumerate(zip(HEX8_POINTS, HEX8_WEIGHTS)):
        _, dN = hex8_shape(xi)
        try:
            _, _, detJ0, detJx_n, grad_X, Fg_n = _kinematics(X, x_n, dN)
            _, _, _, detJx, _, Fg = _kinematics(X, x, dN)
        except np.linalg.LinAlgError:
            return _failed(request, f"singular mapping at Gauss point {g}", FailureKind.FATAL)
        if detJ0 <= _GEOM_TOL:
            return _failed(request, f"nonpositive reference Jacobian at Gauss point {g}", FailureKind.FATAL)
        if detJx_n <= _GEOM_TOL:
            return _failed(request, f"invalid committed Jacobian at Gauss point {g}", FailureKind.FATAL)
        if detJx <= _GEOM_TOL:
            return _failed(request, f"invalid trial Jacobian at Gauss point {g}", FailureKind.RECOVERABLE)
        Jg_n = float(np.linalg.det(Fg_n))
        Jg = float(np.linalg.det(Fg))
        if Jg_n <= 0.0 or Jg <= 0.0:
            kind = FailureKind.FATAL if Jg_n <= 0.0 else FailureKind.RECOVERABLE
            return _failed(request, f"invalid F-bar determinant at Gauss point {g}", kind)
        alpha_n = (Jc_n / Jg_n) ** (1.0 / 3.0)
        alpha = (Jc / Jg) ** (1.0 / 3.0)
        Fbar_n = alpha_n * Fg_n
        Fbar = alpha * Fg
        response = evaluate_material_point(
            request.material,
            MaterialRequest(
                Fbar_n, Fbar, request.material.state_layout.view(request.state_e_n[g]), request.material.properties,
                request.point_properties, request.t_n, request.t_np1, request.need_tangent,
            ),
        )
        if not response.status.ok:
            return _failed(request, response.status.message, response.status.kind)
        trial_state[g] = response.state_trial.values
        P_eff = response.P / (alpha * alpha)
        grad_x_c = grad_X_c @ np.linalg.inv(Fc)
        grad_x_g = grad_X @ np.linalg.inv(Fg)
        q = (grad_x_c - grad_x_g) / 3.0
        B = _B_matrix(grad_x_g)
        sigma = response.P @ Fbar.T / Jc
        dv = detJx * float(weight)
        f += B.T @ _stress_voigt(sigma) * dv
        if K is not None:
            assert response.A_alg is not None
            D = truesdell_voigt(response.A_alg, Fbar, Jc, sigma)
            Bbar = B.copy()
            volumetric = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            for b in range(8):
                for k in range(3):
                    Bbar[:, 3 * b + k] += q[b, k] * volumetric
            K += B.T @ D @ Bbar * dv
            K += np.kron(grad_x_g @ sigma @ grad_x_g.T * dv, np.eye(3))
            sigma_grad = np.einsum("ij,aj->ai", sigma, grad_x_g)
            K -= np.einsum("ai,bk->aibk", sigma_grad, q).reshape(24, 24) * dv
        output.F_raw.append(Fg.copy())
        output.J_raw.append(Jg)
        output.F_material.append(Fbar.copy())
        output.J_material.append(Jc)
        output.P_material.append(response.P.copy())
        output.P_effective.append(P_eff.copy())
        output.cauchy_stress.append(sigma)
    return ElementResponse(f, K, trial_state, EvaluationStatus(), output)


def evaluate_fbar_reference_element(request: ElementRequest) -> ElementResponse:
    """Independent total-reference F-bar kernel used for tangent verification."""
    if request.formulation != "hex8_fbar" or not request.need_tangent:
        raise ValueError("the F-bar reference check requires a tangent-enabled Hex8-Fbar request")
    X = np.asarray(request.X_e, dtype=float)
    x_n = X + request.u_e_n.reshape(8, 3)
    x = X + request.u_e_trial.reshape(8, 3)
    _, dN_c = hex8_shape(np.zeros(3))
    _, _, _, _, grad_X_c, Fc_n = _kinematics(X, x_n, dN_c)
    _, _, _, _, _, Fc = _kinematics(X, x, dN_c)
    Jc_n, Jc = float(np.linalg.det(Fc_n)), float(np.linalg.det(Fc))
    f = np.zeros(24)
    K = np.zeros((24, 24))
    state_trial = np.empty_like(request.state_e_n)
    for g, (xi, weight) in enumerate(zip(HEX8_POINTS, HEX8_WEIGHTS)):
        _, dN = hex8_shape(xi)
        _, _, detJ0, _, grad_X, Fg_n = _kinematics(X, x_n, dN)
        _, _, _, _, _, Fg = _kinematics(X, x, dN)
        Jg_n, Jg = float(np.linalg.det(Fg_n)), float(np.linalg.det(Fg))
        alpha_n = (Jc_n / Jg_n) ** (1.0 / 3.0)
        alpha = (Jc / Jg) ** (1.0 / 3.0)
        response = evaluate_material_point(
            request.material,
            MaterialRequest(
                alpha_n * Fg_n, alpha * Fg, request.material.state_layout.view(request.state_e_n[g]),
                request.material.properties,
                request.point_properties, request.t_n, request.t_np1, True,
            ),
        )
        if not response.status.ok:
            return _failed(request, response.status.message, response.status.kind)
        state_trial[g] = response.state_trial.values
        P_eff = response.P / alpha**2
        dV0 = detJ0 * float(weight)
        f += np.einsum("iI,aI->ai", P_eff, grad_X).ravel() * dV0
        grad_x_c = grad_X_c @ np.linalg.inv(Fc)
        grad_x_g = grad_X @ np.linalg.inv(Fg)
        q = (grad_x_c - grad_x_g) / 3.0
        dFg = np.einsum("ki,bI->bkiI", np.eye(3), grad_X)
        dFbar = alpha * (dFg + q[:, :, None, None] * Fg[None, None, :, :])
        assert response.A_alg is not None
        dPbar = np.einsum("iIjJ,bkjJ->bkiI", response.A_alg, dFbar)
        dPeff = (dPbar - 2.0 * q[:, :, None, None] * response.P) / alpha**2
        K += np.einsum("bkiI,aI->aibk", dPeff, grad_X).reshape(24, 24) * dV0
    return ElementResponse(f, K, state_trial, EvaluationStatus(), None)


def evaluate_element(request: ElementRequest) -> ElementResponse:
    if request.formulation == "hex8_fbar":
        return evaluate_fbar_element(request)
    return evaluate_standard_element(request)
