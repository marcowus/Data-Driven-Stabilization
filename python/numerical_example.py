"""Numerical example matching the MATLAB script using Python tools."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy import signal

from data_driven_stabilization import generate_state_trajectory
from data_driven_stabilization.lmi import solve_dd_lmi, solve_persistent_kernel_test


def three_tank_discretized(Ts: float = 0.1):
    a1 = a2 = a3 = 1.0
    k01 = 0.1
    k12 = 0.5
    k23 = 0.5

    A_c = np.array(
        [
            [-(k01 + k12) / a1, k12 / a1, 0.0],
            [k12 / a2, -k12 / a2, k23 / a2],
            [0.0, 0.0, -k23 / a3],
        ]
    )
    B_c = np.array([[0.0], [1.0 / a2], [0.0]])
    C_c = np.array([[1.0, 0.0, 0.0]])
    D_c = np.array([[0.0]])

    sys_d = signal.cont2discrete((A_c, B_c, C_c, D_c), Ts)
    A, B, _, _, _ = sys_d
    return A, B, Ts


def compute_feedback_gain(Xm: np.ndarray, Xp: np.ndarray, U: np.ndarray):
    pk_result = solve_persistent_kernel_test(U, Xm, Xp)
    r = pk_result.Xm_projected.shape[0]

    Xm_hat = pk_result.Xm_projected
    Xp_hat = pk_result.Xp_projected

    lmi_result = solve_dd_lmi(Xm_hat, Xp_hat)
    if not lmi_result.is_informative:
        raise RuntimeError(f"Dataset is not informative for control (status={lmi_result.status})")
    theta = lmi_result.theta

    if theta is None or theta.size == 0:
        raise RuntimeError("LMI problem returned no feasible theta")

    gain_left = U @ theta
    gain_right = Xm_hat @ theta
    K1 = np.linalg.solve(gain_right.T, gain_left.T).T

    n = Xm.shape[0]
    K2 = np.zeros((K1.shape[0], n - r)) if r < n else np.zeros((K1.shape[0], 0))
    S = pk_result.transformation
    K = np.hstack([K1, K2]) @ S
    return K


def main():
    A, B, Ts = three_tank_discretized()
    n = A.shape[0]
    T = 3

    rng = np.random.default_rng(1)
    U = 5.0 * rng.standard_normal((B.shape[1], T))

    x0 = np.zeros(n)
    Xm, Xp = generate_state_trajectory(A, B, U, x0)

    K = compute_feedback_gain(Xm, Xp, U)

    T_sim = 500
    X_sim = np.zeros((n, T_sim + 1))
    X_sim[:, 0] = np.array([-5.0, -5.0, 2.0])
    U_sim = np.zeros(T_sim)

    for k in range(T_sim):
        U_sim[k] = (K @ X_sim[:, k]).item()
        X_sim[:, k + 1] = A @ X_sim[:, k] + (B @ np.array([U_sim[k]])).ravel()

    time_state = np.arange(T_sim + 1) * Ts
    time_input = np.arange(T_sim) * Ts

    plt.figure(figsize=(8, 4))
    plt.plot(time_state, X_sim[0, :], label=r"$x_1$")
    plt.plot(time_state, X_sim[1, :], label=r"$x_2$")
    plt.plot(time_state, X_sim[2, :], label=r"$x_3$")
    plt.plot(time_input, U_sim, label=r"$u$")
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.ylim([-5, 5])
    plt.xlim([0, 20])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("numerical_example.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
