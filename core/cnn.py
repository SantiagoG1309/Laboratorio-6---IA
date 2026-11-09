"""Red neuronal convolucional (CNN) desde cero con NumPy.

Arquitectura por defecto:
    Entrada 1×H×W → Conv(3×3) → ReLU → MaxPool(2) →
    Conv(3×3) → ReLU → MaxPool(2) → Flatten → Dense → ReLU → Dense → Softmax

El entrenamiento usa entropía cruzada y SGD con Momentum. La convolución se
vectoriza con ``sliding_window_view`` (im2col) para un rendimiento aceptable
en CPU.
"""
from __future__ import annotations

import numpy as np

from core.nn import Dense, ReLU, Momentum, softmax

DTYPE = np.float32


class Conv2D:
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, padding: int = 1, stride: int = 1, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(42)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = int(k)
        self.padding = int(padding)
        self.stride = int(stride)
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = (rng.standard_normal((out_ch, in_ch, k, k)) * scale).astype(DTYPE)
        self.b = np.zeros(out_ch, dtype=DTYPE)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Conv2D espera un tensor (N, C, H, W); recibió {x.shape}")
        n, c, h, w = x.shape
        if c != self.in_ch:
            raise ValueError(f"Conv2D espera {self.in_ch} canales; recibió {c}")
        p, s, k = self.padding, self.stride, self.k
        if h + 2 * p < k or w + 2 * p < k:
            raise ValueError(
                f"Entrada {h}×{w} demasiado pequeña para un kernel {k}×{k} con padding {p}"
            )
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant") if p > 0 else x
        hout = (h + 2 * p - k) // s + 1
        wout = (w + 2 * p - k) // s + 1
        win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))
        win = win[:, :, ::s, ::s][:, :, :hout, :wout]
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * hout * wout, c * k * k)
        out = cols @ self.W.reshape(self.out_ch, -1).T + self.b
        out = out.reshape(n, hout, wout, self.out_ch).transpose(0, 3, 1, 2)
        self._cache = {"cols": cols, "shape": (n, c, h, w), "hout": hout, "wout": wout}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Conv2D.backward() llamado antes de forward()")
        n, c, h, w = self._cache["shape"]
        hout, wout = self._cache["hout"], self._cache["wout"]
        cols = self._cache["cols"]
        dflat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
        self.dW = (dflat.T @ cols).reshape(self.W.shape)
        self.db = dflat.sum(axis=0)
        dcols = dflat @ self.W.reshape(self.out_ch, -1)
        dcols5 = dcols.reshape(n, hout, wout, c, self.k, self.k)
        p, s, k = self.padding, self.stride, self.k
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=DTYPE)
        for ky in range(k):
            for kx in range(k):
                dxp[:, :, ky:ky + hout * s:s, kx:kx + wout * s:s] += dcols5[:, :, :, :, ky, kx].transpose(0, 3, 1, 2)
        if p > 0:
            return dxp[:, :, p:p + h, p:p + w]
        return dxp

    def params(self) -> list[tuple[str, np.ndarray]]:
        return [("W", self.W), ("b", self.b)]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class MaxPool2D:
    def __init__(self, size: int = 2, stride: int = 2):
        self.size = int(size)
        self.stride = int(stride)
        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"MaxPool2D espera un tensor (N, C, H, W); recibió {x.shape}")
        n, c, h, w = x.shape
        k, s = self.size, self.stride
        if h < k or w < k:
            raise ValueError(f"Ventana de pooling {k}×{k} mayor que la entrada {h}×{w}")
        hout = (h - k) // s + 1
        wout = (w - k) // s + 1
        win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
        win = win[:, :, ::s, ::s][:, :, :hout, :wout]
        flat = win.reshape(n, c, hout, wout, k * k)
        idx = flat.argmax(axis=-1)
        out = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
        self._cache = {"idx": idx, "shape": (n, c, h, w), "hout": hout, "wout": wout}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("MaxPool2D.backward() llamado antes de forward()")
        n, c, h, w = self._cache["shape"]
        hout, wout = self._cache["hout"], self._cache["wout"]
        idx = self._cache["idx"]
        k, s = self.size, self.stride
        dx = np.zeros((n, c, h, w), dtype=DTYPE)
        for i in range(k):
            for j in range(k):
                mask = idx == (i * k + j)
                dx[:, :, i:i + hout * s:s, j:j + wout * s:s] += dout * mask
        return dx

    def params(self) -> list[tuple[str, np.ndarray]]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class Flatten:
    def __init__(self):
        self._shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._shape is None:
            raise RuntimeError("Flatten.backward() llamado antes de forward()")
        return dout.reshape(self._shape)

    def params(self) -> list[tuple[str, np.ndarray]]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class CNN:
    def __init__(
        self,
        input_size: int = 48,
        n_out: int = 3,
        filters1: int = 6,
        filters2: int = 12,
        hidden: int = 64,
        seed: int = 42,
        dropout: float = 0.25,
    ):
        rng = np.random.default_rng(seed)
        self.input_size = int(input_size)
        self.n_out = int(n_out)
        self.filters1 = int(filters1)
        self.filters2 = int(filters2)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.layers: list = [
            Conv2D(1, filters1, 3, padding=1, stride=1, rng=rng),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(filters1, filters2, 3, padding=1, stride=1, rng=rng),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
        ]
        feat = filters2 * (input_size // 4) * (input_size // 4)
        self.layers += [Dense(feat, hidden, rng, dropout=dropout), ReLU(), Dense(hidden, n_out, rng)]

    def set_training(self, flag: bool) -> None:
        """Activa o desactiva el dropout de las capas que lo soportan."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = bool(flag)

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=DTYPE)
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        if arr.ndim == 2:
            arr = arr.reshape(-1, 1, self.input_size, self.input_size)
        if arr.ndim != 4 or arr.shape[1:] != (1, self.input_size, self.input_size):
            raise ValueError(
                f"La CNN espera (N, 1, {self.input_size}, {self.input_size}); recibió {arr.shape}"
            )
        for layer in self.layers:
            arr = layer.forward(arr)
        return arr

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.set_training(False)
        return softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.set_training(False)
        return np.argmax(self.forward(x), axis=1)

    def forward_trace(self, x: np.ndarray) -> list[dict]:
        """Forward capa por capa para inspeccionar mapas de activación."""
        self.set_training(False)
        arr = np.asarray(x, dtype=DTYPE)
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        if arr.ndim == 2:
            arr = arr.reshape(-1, 1, self.input_size, self.input_size)
        trace = [{"name": "Entrada", "kind": "input", "array": arr.copy()}]
        conv_i = pool_i = dense_i = 0
        for layer in self.layers:
            arr = layer.forward(arr)
            kind = type(layer).__name__
            if kind == "Conv2D":
                conv_i += 1
                name = f"Conv {conv_i}"
            elif kind == "MaxPool2D":
                pool_i += 1
                name = f"Pool {pool_i}"
            elif kind == "Dense":
                dense_i += 1
                name = f"Densa {dense_i}"
            elif kind == "Flatten":
                name = "Flatten"
            else:
                name = kind
            trace.append({"name": name, "kind": kind, "array": np.asarray(arr).copy()})
        trace.append({"name": "Softmax", "kind": "softmax", "array": softmax(arr)})
        return trace

    def describe_layers(self) -> list[dict]:
        rows: list[dict] = []
        conv_i = pool_i = dense_i = 0
        for layer in self.layers:
            kind = type(layer).__name__
            if kind == "Conv2D":
                conv_i += 1
                rows.append({"layer": f"Conv {conv_i}", "in": layer.in_ch, "out": layer.out_ch,
                             "params": int(layer.W.size + layer.b.size), "kind": kind})
            elif kind == "MaxPool2D":
                pool_i += 1
                rows.append({"layer": f"Pool {pool_i}", "in": None, "out": None, "params": 0, "kind": kind})
            elif kind == "Dense":
                dense_i += 1
                rows.append({"layer": f"Densa {dense_i}", "in": layer.W.shape[0], "out": layer.W.shape[1],
                             "params": int(layer.W.size + layer.b.size), "kind": kind})
            elif kind == "Flatten":
                rows.append({"layer": "Flatten", "in": None, "out": None, "params": 0, "kind": kind})
            else:
                rows.append({"layer": kind, "in": None, "out": None, "params": 0, "kind": kind})
        return rows

    def params(self) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        for layer in self.layers:
            out.extend(layer.params())
        return out

    def grads(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.grads())
        return out

    def loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
        label_smoothing: float = 0.0,
    ) -> tuple[float, list[np.ndarray]]:
        self.set_training(True)
        logits = self.forward(x)
        probs = softmax(logits)
        n = len(y)
        eps = float(np.clip(label_smoothing, 0.0, 0.5))
        if eps > 0.0:
            target = np.full_like(probs, eps / probs.shape[1])
            target[np.arange(n), y] += 1.0 - eps
            loss_vec = -(target * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
            dout = probs - target
        else:
            loss_vec = -np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0))
            dout = probs.copy()
            dout[np.arange(n), y] -= 1.0
        if sample_weights is not None:
            loss_vec = loss_vec * sample_weights
            dout *= sample_weights[:, None]
        loss = float(loss_vec.mean())
        dout = (dout / n).astype(DTYPE)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self.set_training(False)
        return loss, self.grads()


def build_model(
    model_type: str = "cnn",
    input_size: int = 48,
    hidden: int = 64,
    filters1: int = 6,
    filters2: int = 12,
    n_out: int = 3,
    seed: int = 42,
    dropout: float = 0.25,
) -> CNN:
    if str(model_type).lower() != "cnn":
        raise ValueError(f"Arquitectura no soportada: {model_type!r} (solo 'cnn')")
    return CNN(input_size, n_out, filters1, filters2, hidden, seed=seed, dropout=dropout)


def make_optimizer(model, lr: float, momentum: float, weight_decay: float) -> Momentum:
    return Momentum(model.params(), lr=lr, momentum=momentum, weight_decay=weight_decay)
