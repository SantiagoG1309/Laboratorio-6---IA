"""Inspector interactivo de pesos y gradientes del modelo NumPy."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import ACCENT, ACCENT_DARK, BORDER, DANGER, MUTED, MUTED_2, SURFACE_2, TEXT, mix
from ui.widgets.cards import SoftPanel
from ui.widgets.charts import MplCanvas, _style_axes

# Escala divergente coherente con el tema: negativo → cero → positivo.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "aqua_div",
    [DANGER, mix(DANGER, SURFACE_2, 0.62), SURFACE_2, ACCENT_DARK, ACCENT],
)


def parameter_table(model) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Devuelve (nombre legible, valor, gradiente) para cada tensor del modelo."""
    if model is None:
        return []
    names: list[str] = []
    conv_i = 0
    hidden_i = 0
    layers = getattr(model, "layers", [])
    for layer in layers:
        kind = type(layer).__name__
        if kind == "Conv2D":
            conv_i += 1
            base = f"Conv{conv_i}"
        elif kind == "HiddenLayer":
            if layers and layer is layers[-1]:
                base = "Salida"
            else:
                hidden_i += 1
                base = f"Oculta{hidden_i}"
        else:
            continue
        for pname, _ in layer.params():
            names.append(f"{base} · {pname}")
    params = [p for _, p in model.params()]
    grads = list(model.grads())
    if len(grads) < len(params):
        grads += [np.zeros_like(p) for p in params[len(grads):]]
    return [(n, p, g) for n, p, g in zip(names, params, grads)]


def _to_matrix(arr: np.ndarray, limit: int = 420) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if a.size == 0:
        return np.zeros((1, 1))
    a = _block_reduce(a, 0, limit)
    a = _block_reduce(a, 1, limit)
    return a


def _block_reduce(a: np.ndarray, axis: int, limit: int) -> np.ndarray:
    n = a.shape[axis]
    if n <= limit:
        return a
    step = int(np.ceil(n / limit))
    usable = (n // step) * step
    if axis == 0:
        return a[:usable].reshape(usable // step, step, a.shape[1]).mean(axis=1)
    return a[:, :usable].reshape(a.shape[0], usable // step, step).mean(axis=2)


class TensorCanvas(MplCanvas):
    """Mapa de calor de un tensor con escala simétrica centrada en cero."""

    def __init__(self, title: str, colorbar_label: str, parent=None):
        super().__init__(4.0, 3.0)
        self.ax = self.figure.add_subplot(111)
        self._title = title
        self._cbar_label = colorbar_label
        self._empty()

    def _empty(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        _style_axes(self.ax)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(self._title, fontsize=9, loc="left")
        self.ax.text(0.5, 0.5, "Entrena el modelo para inspeccionar", ha="center",
                     va="center", transform=self.ax.transAxes, color=MUTED_2, fontsize=8.5)
        self.draw_idle()

    def update_tensor(self, arr: np.ndarray, subtitle: str = "") -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        _style_axes(self.ax)
        data = _to_matrix(arr)
        vmax = float(np.max(np.abs(data))) if data.size else 0.0
        if vmax <= 1e-12:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title(self._title, fontsize=9, loc="left")
            self.ax.text(0.5, 0.5, "Tensor vacío", ha="center", va="center",
                         transform=self.ax.transAxes, color=MUTED_2, fontsize=8.5)
            self.draw_idle()
            return
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = self.ax.imshow(data, cmap=CMAP_DIV, norm=norm, aspect="auto", interpolation="antialiased")
        cbar = self.figure.colorbar(im, ax=self.ax, fraction=0.038, pad=0.02)
        cbar.ax.tick_params(colors=MUTED, labelsize=6.5, length=2, pad=1)
        cbar.outline.set_edgecolor(BORDER)
        cbar.set_label(self._cbar_label, color=MUTED, fontsize=6.5, labelpad=4)
        self.ax.set_title(self._title, fontsize=9, loc="left")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        if subtitle:
            self.ax.set_xlabel(subtitle, fontsize=7.5, color=MUTED, labelpad=3)
        self.figure.tight_layout(pad=0.4)
        self.draw_idle()


class WeightInspector(QWidget):
    """Lista de tensores + mapas de calor de pesos y gradientes + estadísticas."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._tensors: list[tuple[str, np.ndarray, np.ndarray]] = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        self.list = QListWidget()
        self.list.setFixedWidth(214)
        self.list.setMinimumHeight(300)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setStyleSheet(
            f"QListWidget {{ background: {SURFACE_2}; border: 1px solid {BORDER};"
            "border-radius: 10px; padding: 4px; font-size: 11.5px; }"
        )
        self.list.currentRowChanged.connect(self._select)
        outer.addWidget(self.list, 0)

        right = QVBoxLayout()
        right.setSpacing(8)
        outer.addLayout(right, 1)

        self.stats_host = SoftPanel(padding=10, spacing=6)
        self.stats_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.stats_grid = QHBoxLayout()
        self.stats_grid.setSpacing(6)
        self.stats_host.body().addLayout(self.stats_grid)
        self._stat_labels: dict[str, QLabel] = {}
        for key, label in (
            ("shape", "Forma"),
            ("min", "Mín"),
            ("max", "Máx"),
            ("mean", "Media"),
            ("std", "Desv. típica"),
            ("norm", "Norma L2"),
            ("sparsity", "Ceros"),
        ):
            col = QVBoxLayout()
            col.setSpacing(1)
            cap = QLabel(label)
            cap.setProperty("muted", True)
            cap.setStyleSheet("font-size: 9.5px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {TEXT}; font-size: 11.5px; font-weight: 600;")
            col.addWidget(cap)
            col.addWidget(val)
            self.stats_grid.addLayout(col, 1)
            self._stat_labels[key] = val
        right.addWidget(self.stats_host)

        charts = QHBoxLayout()
        charts.setSpacing(10)
        self.weight_canvas = TensorCanvas("Pesos θ", "valor")
        self.grad_canvas = TensorCanvas("Gradientes ∂L/∂θ", "gradiente")
        self.weight_canvas.setMinimumHeight(300)
        self.grad_canvas.setMinimumHeight(300)
        charts.addWidget(self.weight_canvas, 1)
        charts.addWidget(self.grad_canvas, 1)
        right.addLayout(charts, 1)

        self.note = QLabel(
            "Gradientes calculados con entropía cruzada y softmax sobre el último lote evaluado."
        )
        self.note.setProperty("muted", True)
        self.note.setStyleSheet("font-size: 10px;")
        right.addWidget(self.note)

        self.clear()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._tensors = []
        self.list.blockSignals(True)
        self.list.clear()
        self.list.blockSignals(False)
        self._set_stats(None, None)
        self.weight_canvas._empty()
        self.grad_canvas._empty()
        self.note.setText("Entrena o carga un modelo para inspeccionar sus parámetros.")

    def set_model(self, model) -> None:
        tensors = parameter_table(model)
        if not tensors:
            self.clear()
            return
        self._tensors = tensors
        self.list.blockSignals(True)
        self.list.clear()
        for name, value, _grad in tensors:
            item = QListWidgetItem(f"{name}   {tuple(value.shape)}")
            item.setToolTip(f"{name} · forma {tuple(value.shape)} · {value.size} valores")
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.list.blockSignals(True)
        self.list.setCurrentRow(0)
        self.list.blockSignals(False)
        self._select(0)
        self.note.setText(
            "Gradientes calculados con entropía cruzada y softmax sobre el último lote evaluado."
        )

    # ------------------------------------------------------------------
    def _select(self, row: int) -> None:
        if row < 0 or row >= len(self._tensors):
            return
        name, value, grad = self._tensors[row]
        self.weight_canvas.update_tensor(value, f"{name} · pesos")
        self.grad_canvas.update_tensor(grad, f"{name} · gradiente")
        self._set_stats(value, grad)

    def _set_stats(self, value: np.ndarray | None, grad: np.ndarray | None) -> None:
        if value is None:
            for label in self._stat_labels.values():
                label.setText("—")
            return
        v = np.asarray(value, dtype=np.float64).ravel()
        g = np.asarray(grad, dtype=np.float64).ravel() if grad is not None else np.zeros(1)
        norm = float(np.linalg.norm(g))
        zeros = float((np.abs(v) < 1e-8).mean() * 100) if v.size else 0.0
        self._stat_labels["shape"].setText("×".join(str(s) for s in value.shape))
        self._stat_labels["min"].setText(f"{v.min():+.3g}" if v.size else "—")
        self._stat_labels["max"].setText(f"{v.max():+.3g}" if v.size else "—")
        self._stat_labels["mean"].setText(f"{v.mean():+.3g}" if v.size else "—")
        self._stat_labels["std"].setText(f"{v.std():.3g}" if v.size else "—")
        self._stat_labels["norm"].setText(f"{norm:.3g}")
        self._stat_labels["sparsity"].setText(f"{zeros:.1f}%")
        self._stat_labels["min"].setStyleSheet(
            f"color: {DANGER if v.size and v.min() < 0 else TEXT}; font-size: 11.5px; font-weight: 600;"
        )
        self._stat_labels["max"].setStyleSheet(
            f"color: {ACCENT if v.size and v.max() > 0 else TEXT}; font-size: 11.5px; font-weight: 600;"
        )
