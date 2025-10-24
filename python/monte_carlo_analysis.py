"""Monte Carlo study of data informativity using the Python toolbox."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy import signal

from data_driven_stabilization import analyze_dataset, generate_state_trajectory


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


def main():
    A, B, Ts = three_tank_discretized()
    n = A.shape[0]
    m = B.shape[1]

    T = 50
    N = 100

    rng = np.random.default_rng(0)

    check_id = np.zeros(N)
    check_ddc = np.zeros(N)
    check_ddc_pk = np.zeros(N)

    for j in range(N):
        x0 = rng.poisson(1.0, size=n)
        U = rng.poisson(1.0, size=(m, T))
        Xm, Xp = generate_state_trajectory(A, B, U, x0)
        res_id, res_ddc, res_pk = analyze_dataset(U, Xm, Xp)
        check_id[j] = int(res_id)
        check_ddc[j] = int(res_ddc)
        check_ddc_pk[j] = int(res_pk)

    print(f"Informative for stabilization: {check_ddc_pk.mean() * 100:.1f}%")
    print(f"Informative for control: {check_ddc.mean() * 100:.1f}%")
    print(f"Informative for system identification: {check_id.mean() * 100:.1f}%")

    cumulative = np.arange(1, N + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(cumulative, np.cumsum(check_id) / cumulative, label="Identification")
    plt.plot(cumulative, np.cumsum(check_ddc) / cumulative, label="Control")
    plt.plot(cumulative, np.cumsum(check_ddc_pk) / cumulative, label="Stabilization")
    plt.ylim([0, 1])
    plt.xlabel("Simulation Index")
    plt.ylabel("Informative Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("monte_carlo_informativity.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
