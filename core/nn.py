"""Capas densas, activaciones, funciones de pérdida y optimizadores para la CNN en NumPy puro.

Incluye capas densas (Fully Connected), ReLU, backpropagation, softmax, entropía cruzada y optimización por Momentum.
"""
from __future__ import annotations

import numpy as np

# float32 duplica la velocidad de los productos matriciales de la convolución y
# reduce a la mitad la memoria y el tamaño de los modelos, sin pérdida práctica
# de precisión para esta red.
DTYPE = np.float32


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=DTYPE)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def cross_entropy(probs: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None = None) -> float:
    n = len(y)
    p = np.clip(probs[np.arange(n), y], 1e-12, 1.0)
    loss = -np.log(p)
    if sample_weights is not None:
        loss = loss * sample_weights
    return float(loss.mean())


class Dense:
    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator,
        dropout: float = 0.0,
        dtype: type = DTYPE,
    ):
        limit = np.sqrt(6.0 / (n_in + n_out))
        self.W = rng.uniform(-limit, limit, size=(n_in, n_out)).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.x: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.rng = rng
        self.dropout = float(np.clip(dropout, 0.0, 0.9))
        self.training = False
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = x @ self.W + self.b
        self.mask = None
        if self.training and self.dropout > 0.0:
            # Dropout invertido: se escala durante el entrenamiento para que la
            # inferencia no necesite ningún ajuste.
            keep = 1.0 - self.dropout
            mask = (self.rng.random(out.shape) < keep).astype(out.dtype) / keep
            self.mask = mask
            out = out * mask
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("Dense.backward() llamado antes de forward()")
        if self.mask is not None:
            dout = dout * self.mask
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> list[tuple[str, np.ndarray]]:
        return [("W", self.W), ("b", self.b)]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU:
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward() llamado antes de forward()")
        return dout * self.mask

    def params(self) -> list[tuple[str, np.ndarray]]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class Momentum:
    """SGD con momento: v ← μ·v − lr·∇L ; θ ← θ + v."""

    def __init__(self, params: list[tuple[str, np.ndarray]], lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = float(lr)
        self.mu = float(momentum)
        self.wd = float(weight_decay)
        self.velocities = [np.zeros_like(p, dtype=DTYPE) for _, p in params]

    def step(self, params: list[tuple[str, np.ndarray]], grads: list[np.ndarray]) -> None:
        for i, ((name, p), g) in enumerate(zip(params, grads)):
            g = g.astype(DTYPE)
            if self.wd > 0 and name == "W":
                g = g + self.wd * p
            self.velocities[i] = self.mu * self.velocities[i] - self.lr * g
            p += self.velocities[i]


def gradient_check(target, arg1, arg2=None, eps: float = 1e-6, n: int = 6) -> float:
    """Comprueba gradientes analíticos contra diferencias finitas centrales.

    Acepta dos formas de uso:

    * ``gradient_check(modelo, x, y)``: audita los primeros ``n`` valores de
      cada tensor de parámetros de la red.
    * ``gradient_check(funcion, parametro, gradiente)``: audita una función
      escalar ``f(parametro)`` (la forma que usa el laboratorio de la UI).

    Devuelve el error relativo máximo.
    """
    if callable(target) and not hasattr(target, "loss_and_grads"):
        fn, param, grad = target, np.asarray(arg1), np.asarray(arg2)
        flat = param.reshape(-1)
        gflat = grad.reshape(-1)
        worst = 0.0
        for idx in range(min(n, flat.size)):
            old = float(flat[idx])
            flat[idx] = old + eps
            lp = float(fn(param))
            flat[idx] = old - eps
            lm = float(fn(param))
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            den = max(1e-8, abs(num) + abs(float(gflat[idx])))
            worst = max(worst, abs(num - float(gflat[idx])) / den)
        return float(worst)

    model, x, y = target, arg1, arg2
    _, grads = model.loss_and_grads(x, y)
    params = model.params()
    worst = 0.0
    for (name, p), g in zip(params, grads):
        flat = p.reshape(-1)
        gflat = g.reshape(-1)
        for idx in range(min(n, flat.size)):
            old = float(flat[idx])
            flat[idx] = old + eps
            lp, _ = model.loss_and_grads(x, y)
            flat[idx] = old - eps
            lm, _ = model.loss_and_grads(x, y)
            flat[idx] = old
            num = (lp - lm) / (2 * eps)
            den = max(1e-8, abs(num) + abs(float(gflat[idx])))
            worst = max(worst, abs(num - float(gflat[idx])) / den)
    return float(worst)
