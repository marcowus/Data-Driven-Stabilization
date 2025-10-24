"""Dataset-level informativity checks."""

from __future__ import annotations

import numpy as np
from numpy.linalg import matrix_rank

from .lmi import solve_dd_lmi, solve_persistent_kernel_test


def analyze_dataset(
    Um: np.ndarray,
    Xm: np.ndarray,
    Xp: np.ndarray,
    *,
    rank_tol: float = 1e-9,
) -> tuple[bool, bool, bool]:
    """Evaluate the three informativity conditions used in the MATLAB code."""

    Um = np.asarray(Um, dtype=float)
    Xm = np.asarray(Xm, dtype=float)
    Xp = np.asarray(Xp, dtype=float)

    ddc_result = solve_dd_lmi(Xm, Xp, rank_tol=rank_tol)
    pk_result = solve_persistent_kernel_test(Um, Xm, Xp, rank_tol=rank_tol)

    n, _ = Xm.shape
    m, _ = Um.shape

    stacked_rank = matrix_rank(np.vstack([Xm, Um]), tol=rank_tol)
    check_identification = stacked_rank == n + m

    return check_identification, ddc_result.is_informative, pk_result.is_informative
