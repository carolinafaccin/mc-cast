"""Markov chain of land-class transitions."""
from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power


def transition_matrix(a: np.ndarray, b: np.ndarray, n_classes: int) -> np.ndarray:
    """Row-stochastic matrix P[i, j] = P(class j at t1 | class i at t0).

    `a` and `b` hold class ids 1..n_classes; 0 (no data) is ignored.
    """
    valid = (a > 0) & (b > 0)
    idx = (a[valid].astype(np.int64) - 1) * n_classes + (b[valid].astype(np.int64) - 1)
    counts = np.bincount(idx, minlength=n_classes**2).reshape(n_classes, n_classes).astype(float)
    rows = counts.sum(axis=1, keepdims=True)
    # A class absent at t0 stays where it is.
    P = np.divide(counts, rows, out=np.eye(n_classes), where=rows > 0)
    return P


def rescale(P: np.ndarray, from_years: float, to_years: float) -> np.ndarray:
    """Matrix for a different time step: P**(to_years / from_years)."""
    Q = np.real(fractional_matrix_power(P, to_years / from_years))
    Q = np.clip(Q, 0, None)
    return Q / Q.sum(axis=1, keepdims=True)


def project_counts(counts: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Expected number of cells per class after one application of P."""
    return counts @ P


def class_counts(lulc: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(lulc[lulc > 0].ravel(), minlength=n_classes + 1)[1:].astype(float)
