"""LMI-based informativity checks for data-driven stabilization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import null_space


_LOGGER = logging.getLogger(__name__)


@dataclass
class DDLMIResult:
    """Result container for the direct data-driven LMI."""

    is_informative: bool
    theta: np.ndarray
    status: str


def _matrix_rank(mat: np.ndarray, tol: float = 1e-9) -> int:
    """Compute the numerical rank using a configurable tolerance."""

    return matrix_rank(mat, tol)


def solve_dd_lmi(
    Xm: np.ndarray,
    Xp: np.ndarray,
    *,
    psd_epsilon: float = 1e-6,
    solver: Optional[str] = None,
    solver_kwargs: Optional[dict] = None,
    rank_tol: float = 1e-9,
) -> DDLMIResult:
    """Solve the direct data-driven LMI feasibility problem.

    Parameters
    ----------
    Xm, Xp:
        State data matrices with shapes ``(n, T)``.
    psd_epsilon:
        Minimum eigenvalue enforced on the LMI to provide numerical robustness.
    solver, solver_kwargs:
        Optional overrides for the CVXPY solver and its keyword arguments.
    rank_tol:
        Tolerance for evaluating the rank condition on ``Xm``.
    """

    Xm = np.asarray(Xm, dtype=float)
    Xp = np.asarray(Xp, dtype=float)
    n, T = Xm.shape

    theta = cp.Variable((T, n))

    Xm_theta = Xm @ theta
    Xp_theta = Xp @ theta

    C1 = cp.bmat([[Xm_theta, Xp_theta], [Xp_theta.T, Xm_theta]])
    identity = psd_epsilon * np.eye(2 * n)
    constraints = [C1 >> identity, Xm_theta - Xm_theta.T == 0]

    problem = cp.Problem(cp.Minimize(0), constraints)

    solver_kwargs = dict(solver_kwargs or {})
    if solver is None:
        solver = cp.SCS
    try:
        problem.solve(solver=solver, **solver_kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.warning("CVXPY solver failure: %s", exc)
        return DDLMIResult(False, np.zeros((T, n)), status="error")

    is_informative = (
        problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        and _matrix_rank(Xm, tol=rank_tol) == n
    )

    theta_value = theta.value if theta.value is not None else np.zeros((T, n))
    return DDLMIResult(is_informative, theta_value, status=problem.status)


@dataclass
class PersistentKernelResult:
    """Result container for the persistent-kernel test."""

    is_informative: bool
    transformation: np.ndarray
    Xm_projected: np.ndarray
    Xp_projected: np.ndarray


def solve_persistent_kernel_test(
    Um: np.ndarray,
    Xm: np.ndarray,
    Xp: np.ndarray,
    *,
    kernel_tol: float = 1e-8,
    rank_tol: float = 1e-9,
) -> PersistentKernelResult:
    """Evaluate the persistent-kernel stabilization test."""

    Um = np.asarray(Um, dtype=float)
    Xm = np.asarray(Xm, dtype=float)
    Xp = np.asarray(Xp, dtype=float)

    n, T = Xm.shape
    m, _ = Um.shape

    r = _matrix_rank(Xm, tol=rank_tol)

    if r < n:
        U, s, Vh = np.linalg.svd(Xm, full_matrices=True)
        S = U.T
        SXm = S @ Xm
        SXp = S @ Xp

        Xm_hat = SXm[:r, :]
        Xp_hat = SXp[:r, :]

        stacked = np.vstack([Xm_hat, Um])
        rank_stacked = _matrix_rank(stacked, tol=rank_tol)

        kernel = null_space(Xm.T)
        kernel_projection = kernel.T @ Xp if kernel.size else np.zeros((0, T))
        kernel_norm = np.linalg.norm(kernel_projection)

        is_informative = kernel_norm <= kernel_tol and rank_stacked == r + m

        return PersistentKernelResult(is_informative, S, Xm_hat, Xp_hat)

    # Full-rank case: fallback to direct LMI test
    lmi_result = solve_dd_lmi(Xm, Xp, rank_tol=rank_tol)
    identity = np.eye(n)
    Xm_hat = Xm.copy()
    Xp_hat = Xp.copy()
    return PersistentKernelResult(lmi_result.is_informative, identity, Xm_hat, Xp_hat)
