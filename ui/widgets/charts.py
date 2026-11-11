"""Gráficos científicos embebidos en Qt con Matplotlib."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

from core.config import CLASS_COLORS, CLASS_LABELS, CLASSES
from ui.theme import (
    ACCENT,
    ACCENT_2,
    ACCENT_DARK,
    BG_ALT,
    BORDER,
    CHART_GRID,
    DANGER,
    FONT_FAMILY,
    MUTED,
    MUTED_2,
    ON_ACCENT,
    PURPLE,
    SUCCESS,
    SURFACE,
    TEXT,
    WARNING,
    mix,
)

rcParams["font.family"] = [FONT_FAMILY, "DejaVu Sans", "sans-serif"]
rcParams["font.size"] = 8.5
rcParams["axes.edgecolor"] = BORDER
rcParams["axes.labelcolor"] = MUTED
rcParams["xtick.color"] = MUTED
rcParams["ytick.color"] = MUTED
rcParams["text.color"] = TEXT
rcParams["figure.facecolor"] = SURFACE
rcParams["axes.facecolor"] = SURFACE

# Escala propia para la matriz de confusión: fondo profundo → acento intenso.
CMAP_AQUA = LinearSegmentedColormap.from_list(
    "aquacnn",
    [BG_ALT, mix(ACCENT_DARK, BG_ALT, 0.35), ACCENT_DARK, ACCENT],
)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.grid(True, color=CHART_GRID, linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
    ax.title.set_color(TEXT)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 5, height: float = 3, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE, tight_layout=True)
        super().__init__(self.figure)
        self.setStyleSheet(f"background: {SURFACE};")


class HistogramCanvas(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(5, 2.4)
        self.ax = self.figure.add_subplot(111)
        _style_axes(self.ax)
        self.ax.set_xlim(0, 255)
        self.ax.set_xlabel("Intensidad de píxel")
        self.ax.set_ylabel("Frecuencia")
        self.ax.set_title("Histograma de intensidad", fontsize=9, loc="left")
        self._before = None
        self._after = None
        self._draw_empty()

    def _draw_empty(self) -> None:
        self.ax.clear()
        _style_axes(self.ax)
        self.ax.set_xlim(0, 255)
        self.ax.set_title("Histograma de intensidad", fontsize=9, loc="left")
        self.ax.text(0.5, 0.5, "Selecciona una imagen", ha="center", va="center",
                     transform=self.ax.transAxes, color=MUTED_2, fontsize=9)
        self.draw_idle()

    def update_hist(self, before: np.ndarray, after: np.ndarray, equalization: str = "") -> None:
        self.ax.clear()
        _style_axes(self.ax)
        x = np.arange(256)
        curves: list[np.ndarray] = []
        if before is not None and len(before) >= 256:
            b = np.asarray(before[:256], dtype=float)
            total = float(b.sum()) or 1.0
            b = b / total
            curves.append(b)
            self.ax.fill_between(x, b, color=mix(SURFACE, ACCENT, 0.22), alpha=0.55, linewidth=0)
            self.ax.plot(x, b, color=mix(MUTED, ACCENT, 0.35), linewidth=1.0, alpha=0.85, label="Original")
        if after is not None and len(after) >= 256:
            a = np.asarray(after[:256], dtype=float)
            total = float(a.sum()) or 1.0
            a = a / total
            curves.append(a)
            self.ax.plot(x, a, color=ACCENT, linewidth=1.35, label="Procesado")
        if curves:
            # La ecualización aplana el histograma: se recorta el eje Y al
            # percentil 99 para que ambas curvas sigan siendo comparables.
            peak = max(float(np.percentile(c, 99)) for c in curves)
            top = max(peak * 1.25, 1e-4)
            self.ax.set_ylim(0, top)
        self.ax.set_xlim(0, 255)
        self.ax.set_xlabel("Intensidad de píxel")
        self.ax.set_ylabel("Frecuencia relativa")
        title = "Histograma de intensidad"
        if equalization:
            short = equalization.split("(")[0].strip()
            title = f"Histograma · {short}"
        self.ax.set_title(title, fontsize=9, loc="left")
        if curves:
            self.ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=MUTED)
        self.figure.tight_layout()
        self.draw_idle()


class TrainingChart(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(6, 4.2)
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_acc = self.figure.add_subplot(212)
        for ax in (self.ax_loss, self.ax_acc):
            _style_axes(ax)
        self._reset()

    def _reset(self) -> None:
        for ax in (self.ax_loss, self.ax_acc):
            ax.clear()
            _style_axes(ax)
        self.ax_loss.set_title("Pérdida (entropía cruzada)", fontsize=9, loc="left")
        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_title("Exactitud por época", fontsize=9, loc="left")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_xlabel("Época")
        self.ax_acc.set_ylim(0, 100)
        for ax in (self.ax_loss, self.ax_acc):
            ax.text(0.5, 0.5, "Sin entrenamiento activo", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED_2, fontsize=9)
        self.figure.tight_layout()
        self.draw_idle()

    def update_history(self, history: dict[str, list[float]], best_epoch: int = 0) -> None:
        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])
        if not loss:
            self._reset()
            return
        epochs = np.arange(1, len(loss) + 1)
        self.ax_loss.clear()
        self.ax_acc.clear()
        _style_axes(self.ax_loss)
        _style_axes(self.ax_acc)
        self.ax_loss.plot(epochs, loss, color=ACCENT, linewidth=1.6, label="Entrenamiento")
        if val_loss and len(val_loss) == len(loss):
            self.ax_loss.plot(epochs, val_loss, color=WARNING, linewidth=1.4,
                              marker="o", markersize=2.5, label="Validación (instantáneas)")
        self.ax_loss.set_title("Pérdida (entropía cruzada)", fontsize=9, loc="left")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=MUTED)
        self.ax_acc.plot(epochs, np.array(acc) * 100, color=SUCCESS, linewidth=1.6, label="Entrenamiento")
        if val_acc and len(val_acc) == len(loss):
            self.ax_acc.plot(epochs, np.array(val_acc) * 100, color=PURPLE, linewidth=1.4,
                             marker="o", markersize=2.5, label="Validación (instantáneas)")
        self.ax_acc.set_title("Exactitud por época", fontsize=9, loc="left")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_xlabel("Época")
        self.ax_acc.set_ylim(0, 100)
        self.ax_acc.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=MUTED)
        for ax in (self.ax_loss, self.ax_acc):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if best_epoch:
            for ax in (self.ax_loss, self.ax_acc):
                ax.axvline(best_epoch, color=ACCENT_2, linewidth=0.9, linestyle="--", alpha=0.7)
        self.figure.tight_layout()
        self.draw_idle()


class ConfusionMatrixCanvas(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(4.2, 3.8)
        self.ax = self.figure.add_subplot(111)
        self._empty()

    def _empty(self) -> None:
        self.ax.clear()
        _style_axes(self.ax)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, "Entrena un modelo para ver la matriz", ha="center", va="center",
                     transform=self.ax.transAxes, color=MUTED_2, fontsize=9)
        self.draw_idle()

    def update_cm(self, cm: np.ndarray, labels: list[str] | None = None) -> None:
        labels = labels or [CLASS_LABELS[c] for c in CLASSES]
        self.ax.clear()
        self.ax.set_facecolor(SURFACE)
        cm = np.asarray(cm)
        if cm.size == 0:
            self._empty()
            return
        im = self.ax.imshow(cm, cmap=CMAP_AQUA, aspect="auto", vmin=0, vmax=max(1, int(cm.max())))
        n = cm.shape[0]
        vmax = max(1, int(cm.max()))
        for i in range(n):
            for j in range(n):
                val = int(cm[i, j])
                if val == 0:
                    color = MUTED_2
                elif val > vmax * 0.55:
                    color = ON_ACCENT
                else:
                    color = TEXT
                self.ax.text(j, i, str(val), ha="center", va="center", color=color,
                             fontsize=10, fontweight="bold")
        self.ax.set_xticks(range(n))
        self.ax.set_yticks(range(n))
        self.ax.set_xticklabels(labels, rotation=0, fontsize=8)
        self.ax.set_yticklabels(labels, fontsize=8)
        self.ax.set_xlabel("Predicción", fontsize=8)
        self.ax.set_ylabel("Real", fontsize=8)
        self.ax.set_title("Matriz de confusión · conjunto de prueba", fontsize=9, loc="left")
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.figure.tight_layout()
        self.draw_idle()


class ClassBarsCanvas(MplCanvas):
    def __init__(self, parent=None):
        super().__init__(4.6, 2.1)
        self.ax = self.figure.add_subplot(111)
        self._empty()

    def _empty(self) -> None:
        self.ax.clear()
        _style_axes(self.ax)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                     transform=self.ax.transAxes, color=MUTED_2, fontsize=9)
        self.draw_idle()

    def update_counts(self, counts: dict[str, int]) -> None:
        if not counts or sum(counts.values()) == 0:
            self._empty()
            return
        labels = [CLASS_LABELS[c] for c in CLASSES]
        values = [counts.get(c, 0) for c in CLASSES]
        colors = [CLASS_COLORS[c] for c in CLASSES]
        self.ax.clear()
        _style_axes(self.ax)
        y = np.arange(len(labels))
        self.ax.barh(y, values, color=colors, alpha=0.85, height=0.55)
        for yi, v in zip(y, values):
            self.ax.text(v + max(values) * 0.02, yi, str(v), va="center", color=TEXT, fontsize=8)
        self.ax.set_yticks(y)
        self.ax.set_yticklabels(labels, fontsize=8)
        self.ax.set_xlim(0, max(values) * 1.18)
        self.ax.set_xlabel("Imágenes", fontsize=8)
        self.ax.grid(True, axis="x", color=CHART_GRID, linewidth=0.6, alpha=0.7)
        self.ax.grid(False, axis="y")
        self.ax.set_title("Distribución por clase", fontsize=9, loc="left")
        self.figure.tight_layout()
        self.draw_idle()
