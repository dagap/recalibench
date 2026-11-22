"""Calibrators for choice tasks (identity and temperature live in binary.py, serving both).

    vector scaling  Z = w * log P + b             (Guo et al. 2017): per-class scale and bias
    dirichlet       Z = W log P + b               (Kull et al. 2019): full linear map on log P,
                    off-diagonal and intercept L2 (ODIR); strength by 3-fold CV on the fit data

Per-class terms can correct a model that systematically favours one option,
e.g. Jev "leans toward the option that comes first" (TypeSafe docs).
"""

from typing import Any

import numpy as np

from recalibench.calibrators._fit import finish, lbfgs, safe_log, softmax, softmax_nll
from recalibench.core.protocols import FloatArray, IntArray


class VectorScaling:
    supports = frozenset({"multiclass"})
    parametric = True

    def __init__(self, name: str, l2: float = 1e-4) -> None:
        self.name = name
        self.l2 = l2
        self.w: FloatArray = np.array([])
        self.b: FloatArray = np.array([])

    def fit(self, P: FloatArray, y: IntArray) -> None:
        L = safe_log(P)
        k, lam = L.shape[1], self.l2

        def objective(theta: FloatArray) -> tuple[float, FloatArray]:
            w, b = theta[:k], theta[k:]
            loss, G = softmax_nll(L * w + b, y)
            penalty = lam * float(((w - 1) ** 2).sum() + (b**2).sum())
            grad = np.concatenate(
                [(G * L).sum(axis=0) + 2 * lam * (w - 1), G.sum(axis=0) + 2 * lam * b]
            )
            return loss + penalty, grad

        theta = lbfgs(objective, np.concatenate([np.ones(k), np.zeros(k)]))
        self.w, self.b = theta[:k], theta[k:]

    def predict(self, P: FloatArray) -> FloatArray:
        return finish(softmax(safe_log(P) * self.w + self.b))

    def params(self) -> dict[str, Any]:
        return {"w": self.w.tolist(), "b": self.b.tolist()}


def _fit_dirichlet(L: FloatArray, y: IntArray, lam: float) -> tuple[FloatArray, FloatArray]:
    k = L.shape[1]
    off = 1.0 - np.eye(k)

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        W, b = theta[: k * k].reshape(k, k), theta[k * k :]
        loss, G = softmax_nll(L @ W.T + b, y)
        penalty = lam * float(((W * off) ** 2).sum() + (b**2).sum())
        gW = G.T @ L + 2 * lam * W * off
        gb = G.sum(axis=0) + 2 * lam * b
        return loss + penalty, np.concatenate([gW.ravel(), gb])

    theta = lbfgs(objective, np.concatenate([np.eye(k).ravel(), np.zeros(k)]), max_iter=1000)
    return theta[: k * k].reshape(k, k), theta[k * k :]


class Dirichlet:
    supports = frozenset({"multiclass"})
    parametric = True

    def __init__(
        self,
        name: str,
        lambdas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        cv_folds: int = 3,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.lambdas: tuple[float, ...] = tuple(float(x) for x in lambdas)
        if not self.lambdas:
            raise ValueError("dirichlet needs at least one lambda")
        self.cv_folds = cv_folds
        self.seed = seed
        self.lam = self.lambdas[0]
        self.W: FloatArray = np.array([])
        self.b: FloatArray = np.array([])

    def _choose_lambda(self, L: FloatArray, y: IntArray) -> float:
        rng = np.random.default_rng(self.seed)
        folds = rng.permutation(len(y)) % self.cv_folds
        best, best_loss = self.lambdas[0], np.inf
        for lam in self.lambdas:
            losses: list[float] = []
            for f in range(self.cv_folds):
                tr, va = folds != f, folds == f
                W, b = _fit_dirichlet(L[tr], y[tr], lam)
                losses.append(softmax_nll(L[va] @ W.T + b, y[va])[0])
            mean = float(np.mean(losses))
            if mean < best_loss:
                best, best_loss = lam, mean
        return best

    def fit(self, P: FloatArray, y: IntArray) -> None:
        L = safe_log(P)
        self.lam = self._choose_lambda(L, y) if len(self.lambdas) > 1 else self.lam
        self.W, self.b = _fit_dirichlet(L, y, self.lam)

    def predict(self, P: FloatArray) -> FloatArray:
        return finish(softmax(safe_log(P) @ self.W.T + self.b))

    def params(self) -> dict[str, Any]:
        return {"lambda": self.lam, "W": self.W.tolist(), "b": self.b.tolist()}
