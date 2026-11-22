"""Shared numerics for parametric calibrators: softmax NLL with analytic gradients.

Every parametric calibrator here maps log-probabilities L (n, k) to new logits
Z = f(L; theta) and minimises mean softmax cross-entropy (plus a penalty) with
L-BFGS. Binary calibrators use k = 2 columns [yes, no], for which softmax over
Z reduces to the usual sigmoid of a logit difference.

scipy's optimiser is untyped under pyright strict; it is only called here.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.optimize import (  # pyright: ignore[reportMissingTypeStubs]
    minimize,  # pyright: ignore[reportUnknownVariableType]
    minimize_scalar,  # pyright: ignore[reportUnknownVariableType]
)

from recalibench.core.protocols import FloatArray, IntArray

CLIP = 1e-6

Objective = Callable[[FloatArray], tuple[float, FloatArray]]


def safe_log(P: FloatArray) -> FloatArray:
    return np.log(np.clip(P, CLIP, 1.0))


def softmax(Z: FloatArray) -> FloatArray:
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def finish(P: FloatArray) -> FloatArray:
    """Clip to [CLIP, 1 - CLIP] and renormalise each row."""
    P = np.clip(P, CLIP, 1 - CLIP)
    return P / P.sum(axis=1, keepdims=True)


def softmax_nll(Z: FloatArray, y: IntArray) -> tuple[float, FloatArray]:
    """Mean cross-entropy of softmax(Z) against class indices y, and dLoss/dZ."""
    n = Z.shape[0]
    shifted = Z - Z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    loss = float((log_norm - shifted[np.arange(n), y]).mean())
    grad = softmax(Z)
    grad[np.arange(n), y] -= 1.0
    return loss, grad / n


def lbfgs(objective: Objective, theta0: FloatArray, max_iter: int = 500) -> FloatArray:
    result = cast(
        Any,
        minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter}),
    )
    return np.asarray(result.x, dtype=np.float64)


def fit_temperature(L: FloatArray, y: IntArray) -> float:
    """T minimising NLL of softmax(L / T), searched over log T in [-5, 5]."""

    def nll(log_t: float) -> float:
        return softmax_nll(L / np.exp(log_t), y)[0]

    result = cast(
        Any, minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded", options={"xatol": 1e-6})
    )
    return float(np.exp(float(result.x)))


def binary_logistic(
    X: FloatArray, y: IntArray, l2: float = 1e-6, max_iter: int = 500
) -> tuple[FloatArray, float]:
    """Logistic regression of [class 0 = yes] on features X; returns (weights, bias).

    Written as a 2-class softmax with logits [X w + c, 0], so it shares softmax_nll.
    """
    d = X.shape[1]

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        w, c = theta[:d], theta[d]
        Z = np.column_stack([X @ w + c, np.zeros(len(X))])
        loss, G = softmax_nll(Z, y)
        g0 = G[:, 0]
        grad = np.concatenate([X.T @ g0 + 2 * l2 * w, [g0.sum()]])
        return loss + l2 * float(w @ w), grad

    theta = lbfgs(objective, np.concatenate([np.ones(d), [0.0]]), max_iter)
    return theta[:d], float(theta[d])
