"""Utilities for generating discrete-time state trajectories."""

from __future__ import annotations

import numpy as np


def generate_state_trajectory(A: np.ndarray, B: np.ndarray, U: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the discrete-time system ``x_{k+1} = A x_k + B u_k``.

    Parameters
    ----------
    A, B : np.ndarray
        Discrete-time system matrices with shapes ``(n, n)`` and ``(n, m)``.
    U : np.ndarray
        Input sequence with shape ``(m, T)``.
    x0 : np.ndarray
        Initial state vector with shape ``(n,)`` or ``(n, 1)``.

    Returns
    -------
    Xm : np.ndarray
        Matrix collecting the state trajectory at times ``0`` through ``T-1``.
    Xp : np.ndarray
        Matrix collecting the state trajectory at times ``1`` through ``T``.
    """

    A = np.asarray(A)
    B = np.asarray(B)
    U = np.asarray(U)
    x0 = np.asarray(x0).reshape(-1)

    n, m = B.shape
    _, T = U.shape

    x = np.zeros((n, T + 1), dtype=float)
    x[:, 0] = x0

    for k in range(T):
        x[:, k + 1] = A @ x[:, k] + B @ U[:, k]

    Xm = x[:, :T]
    Xp = x[:, 1:]
    return Xm, Xp
