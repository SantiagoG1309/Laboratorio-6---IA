"""Diagrama animado de la red neuronal: círculos, líneas y mapas de activación."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QToolTip, QWidget

from core.config import CLASS_COLORS, CLASS_LABELS, CLASSES
from ui import preferences
from ui.theme import ACCENT, ACCENT_2, BG_ALT, BORDER, MUTED, PURPLE, SUCCESS, SURFACE_2, TEXT, WARNING, mix, qcolor


def _kind(stage: str) -> str:
    s = stage.lower()
    if s.startswith("entrada") or s.startswith("imagen"):
        return "input"
    if s.startswith("conv"):
        return "conv"
    if s.startswith("relu"):
        return "relu"
    if s.startswith("pool"):
        return "pool"
    if s.startswith("flatten"):
        return "flatten"
    if s.startswith("densa") or s.startswith("fc"):
        return "dense"
    if s.startswith("softmax"):
        return "softmax"
    return "other"


def _shape_list(shape) -> list[int]:
    if not shape:
        return []
    return [int(v) for v in shape]


class NNDiagram(QWidget):
    """Dibuja la arquitectura completa con nodos, conexiones y activaciones."""

    layerClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(330)
        self.setMinimumWidth(760)
        self.setMouseTracking(True)
        self._rows: list[dict] = []
        self._rects: list[QRectF] = []
        self._hover = -1
        self._phase = 0.0
        self._image: np.ndarray | None = None
        self._activations: list[dict] = []
        self._model = None
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if preferences.get("animations", True) and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    # ------------------------------------------------------------------
    def set_architecture(self, rows: list) -> None:
        """``rows`` puede ser una lista de ``ArchRow`` o de diccionarios."""
        out: list[dict] = []
        for r in rows:
            if hasattr(r, "to_dict"):
                d = r.to_dict()
                d["output_shape"] = tuple(r.output_shape or ())
                d["params_int"] = int(r.params)
                d["stage"] = r.stage
                d["operation"] = r.operation
            else:
                d = dict(r)
            d["kind"] = _kind(d.get("stage", ""))
            out.append(d)
        self._rows = out
        self.update()

    def set_image(self, image: np.ndarray | None) -> None:
        self._image = image
        self.update()

    def set_model(self, model) -> None:
        self._model = model
        self.update()

    def set_activations(self, trace: list[dict] | None) -> None:
        self._activations = list(trace or [])
        self.update()

    def clear_activations(self) -> None:
        self._activations = []
        self.update()

    def _tick(self) -> None:
        if not self.isVisible():
            self._timer.stop()
            return
        if not preferences.get("animations", True):
            self._timer.stop()
            return
        self._phase = (self._phase + 0.035) % 1.0
        self.update()

    # ------------------------------------------------------------------
    def _layout(self) -> None:
        self._rects = []
        n = max(1, len(self._rows))
        margin = 26
        top = 58
        bottom = 44
        width = max(320, self.width() - 2 * margin)
        bw = width / n
        h = max(120, self.height() - top - bottom)
        for i in range(n):
            self._rects.append(QRectF(margin + i * bw + 3, top, max(24, bw - 6), h))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout()

    def _activation_for(self, index: int) -> np.ndarray | None:
        if not self._activations:
            return None
        if index < len(self._activations):
            arr = self._activations[index].get("array")
            return np.asarray(arr) if arr is not None else None
        if self._activations:
            arr = self._activations[-1].get("array")
            return np.asarray(arr) if arr is not None else None
        return None

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(BG_ALT))
        self._layout()
        if not self._rows:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Configura la red para ver su arquitectura")
            p.end()
            return

        # Rejilla de fondo
        p.setPen(QPen(qcolor(BORDER, 70), 1))
        step = 26
        for x in range(0, self.width(), step):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            p.drawLine(0, y, self.width(), y)

        # Título
        p.setPen(QColor(TEXT))
        f = QFont("Segoe UI", 10)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.drawText(18, 24, "Arquitectura de la red · círculos = neuronas · líneas = conexiones")

        # Conexiones primero (quedan detrás de los nodos)
        for i in range(len(self._rows) - 1):
            self._draw_connection(p, i, i + 1)

        # Bloques
        for i, row in enumerate(self._rows):
            self._draw_block(p, i, row, self._rects[i], i == self._hover)

        # Leyenda inferior
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(MUTED))
        p.drawText(
            18, self.height() - 16,
            "Pasa el cursor sobre una capa para ver su operación · clic para inspeccionar sus activaciones",
        )
        p.end()

    # ------------------------------------------------------------------
    def _draw_connection(self, p: QPainter, i: int, j: int) -> None:
        a, b = self._rects[i], self._rects[j]
        y = a.center().y()
        x1, x2 = a.right(), b.left()
        if x2 <= x1:
            return
        grad = QLinearGradient(x1, y, x2, y)
        grad.setColorAt(0.0, qcolor(ACCENT, 60))
        grad.setColorAt(1.0, qcolor(ACCENT_2, 90))
        p.setPen(QPen(QBrush(grad), 2.0))
        p.drawLine(QPointF(x1, y), QPointF(x2 - 7, y))
        # punta de flecha
        path = QPainterPath()
        path.moveTo(x2 - 2, y)
        path.lineTo(x2 - 9, y - 4.5)
        path.lineTo(x2 - 9, y + 4.5)
        path.closeSubpath()
        p.setBrush(qcolor(ACCENT_2, 170))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)
        # pulso animado
        for k in range(3):
            t = (self._phase + k / 3.0) % 1.0
            x = x1 + (x2 - x1) * t
            alpha = int(200 * math.sin(math.pi * t))
            p.setBrush(qcolor(TEXT, max(0, alpha)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), 2.6, 2.6)

    def _draw_block(self, p: QPainter, index: int, row: dict, rect: QRectF, hover: bool) -> None:
        kind = row.get("kind", "other")
        color = QColor(row.get("color", ACCENT))
        if hover:
            p.setBrush(qcolor(ACCENT, 16))
            p.setPen(QPen(qcolor(ACCENT, 120), 1.2))
            p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 9, 9)

        cx = rect.center().x()
        cy = rect.center().y()
        act = self._activation_for(index)

        if kind == "input":
            side = min(rect.width() * 0.62, rect.height() * 0.5, 76)
            img_rect = QRectF(cx - side / 2, cy - side / 2 - 8, side, side)
            if self._image is not None:
                self._draw_array(p, self._image, img_rect)
            else:
                p.setPen(QPen(qcolor(ACCENT_2, 120), 1))
                p.setBrush(qcolor(mix(BG_ALT, ACCENT, 0.35)))
                p.drawRoundedRect(img_rect, 5, 5)
                p.setPen(qcolor(ACCENT_2, 90))
                for k in range(1, 6):
                    p.drawLine(img_rect.left() + k * side / 6, img_rect.top(),
                               img_rect.left() + k * side / 6, img_rect.bottom())
                    p.drawLine(img_rect.left(), img_rect.top() + k * side / 6,
                               img_rect.right(), img_rect.top() + k * side / 6)
            self._caption(p, rect, row, "Entrada", color)
            return

        if kind in ("conv", "pool"):
            n_planes = 3 if kind == "conv" else 2
            base = min(rect.width() * 0.55, rect.height() * 0.36, 52)
            shrink = 0.78 if kind == "pool" else 1.0
            for k in range(n_planes - 1, -1, -1):
                off = k * 5
                plane = QRectF(
                    cx - base * shrink / 2 + off,
                    cy - base * shrink / 2 - off + 2,
                    base * shrink,
                    base * shrink,
                )
                if act is not None and act.ndim >= 3 and act.shape[0] > 0:
                    self._draw_array(p, act[0], plane)
                else:
                    p.setBrush(QColor(color.red(), color.green(), color.blue(), 34 + 14 * k))
                    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 170), 1.1))
                    p.drawRoundedRect(plane, 4, 4)
                    p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 60), 1))
                    for g in range(1, 4):
                        p.drawLine(plane.left() + g * plane.width() / 4, plane.top(),
                                   plane.left() + g * plane.width() / 4, plane.bottom())
                        p.drawLine(plane.left(), plane.top() + g * plane.height() / 4,
                                   plane.right(), plane.top() + g * plane.height() / 4)
            self._caption(p, rect, row, row.get("stage", kind), color)
            return

        if kind == "flatten":
            bar_w = min(rect.width() * 0.5, 34)
            bar_h = min(rect.height() * 0.52, 130)
            bar = QRectF(cx - bar_w / 2, cy - bar_h / 2, bar_w, bar_h)
            p.setBrush(qcolor(PURPLE, 26))
            p.setPen(QPen(qcolor(PURPLE, 160), 1.1))
            p.drawRoundedRect(bar, 5, 5)
            cells = 14
            for k in range(cells):
                y = bar.top() + (k + 0.5) * bar.height() / cells
                val = 0.5
                if act is not None and act.size:
                    flat = np.asarray(act).ravel()
                    val = float(flat[int(k * flat.size / cells)]) if flat.size else 0.0
                    lo, hi = float(flat.min()), float(flat.max())
                    val = (val - lo) / (hi - lo) if hi > lo else 0.5
                c = QColor.fromHsvF(0.56 - 0.45 * float(np.clip(val, 0, 1)), 0.75, 0.55 + 0.4 * float(np.clip(val, 0, 1)))
                p.setBrush(c)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(QRectF(bar.left() + 3, y - 2.2, bar_w - 6, 3.6), 1.5, 1.5)
            self._caption(p, rect, row, "Flatten", QColor(PURPLE))
            return

        if kind in ("dense", "softmax"):
            out = _shape_list(row.get("output_shape"))
            units = out[-1] if out else 3
            is_out = kind == "softmax" or units <= 6
            shown = min(units, 7 if not is_out else units)
            r = min(8.0, max(3.5, rect.width() * 0.11))
            gap = min(22.0, (rect.height() * 0.62) / max(1, shown))
            top = cy - (shown - 1) * gap / 2
            vals = None
            if act is not None:
                a = np.asarray(act).ravel()
                vals = a[:shown] if a.size >= shown else None
            for k in range(shown):
                yy = top + k * gap
                if vals is not None:
                    v = float(vals[k])
                    lo, hi = float(np.min(vals)), float(np.max(vals))
                    norm = (v - lo) / (hi - lo) if hi > lo else 0.5
                    c = QColor(color.red(), color.green(), color.blue(), int(90 + 150 * norm))
                else:
                    c = QColor(color.red(), color.green(), color.blue(), 150)
                p.setBrush(qcolor(BG_ALT))
                p.setPen(QPen(c, 1.8))
                p.drawEllipse(QPointF(cx, yy), r, r)
                p.setBrush(c)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(cx, yy), max(1.4, r * 0.38), max(1.4, r * 0.38))
            if units > shown:
                p.setPen(QColor(MUTED))
                p.setFont(QFont("Segoe UI", 8))
                p.drawText(QRectF(rect.left(), top + shown * gap - 4, rect.width(), 14),
                           Qt.AlignmentFlag.AlignCenter, f"+{units - shown}")
            if is_out:
                for k, cls in enumerate(CLASSES[:units]):
                    yy = top + k * gap + r + 8
                    p.setPen(QColor(CLASS_COLORS[cls]))
                    p.setFont(QFont("Segoe UI", 7))
                    p.drawText(QRectF(rect.left(), yy - 7, rect.width(), 12),
                               Qt.AlignmentFlag.AlignCenter, CLASS_LABELS[cls][:7])
            self._caption(p, rect, row, row.get("stage", "Densa"), color)
            return

        if kind == "relu":
            r = QRectF(cx - min(rect.width() * 0.26, 22), cy - min(rect.height() * 0.16, 26),
                       min(rect.width() * 0.52, 44), min(rect.height() * 0.32, 52))
            p.setBrush(qcolor(ACCENT_2, 22))
            p.setPen(QPen(qcolor(ACCENT_2, 120), 1))
            p.drawRoundedRect(r, 6, 6)
            path = QPainterPath()
            path.moveTo(r.left() + 5, r.bottom() - 8)
            path.lineTo(r.center().x(), r.bottom() - 8)
            path.lineTo(r.right() - 5, r.top() + 8)
            p.setPen(QPen(qcolor(ACCENT_2, 220), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            self._caption(p, rect, row, "ReLU", QColor(ACCENT_2))
            return

        self._caption(p, rect, row, row.get("stage", "?"), color)

    # ------------------------------------------------------------------
    def _caption(self, p: QPainter, rect: QRectF, row: dict, title: str, color: QColor) -> None:
        p.setPen(color)
        f = QFont("Segoe UI", 8)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.drawText(QRectF(rect.left(), rect.bottom() - 30, rect.width(), 14),
                   Qt.AlignmentFlag.AlignCenter, title)
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 7))
        out = row.get("output") or row.get("output_shape") or ""
        if isinstance(out, (tuple, list)):
            out = " × ".join(str(v) for v in out)
        p.drawText(QRectF(rect.left(), rect.bottom() - 16, rect.width(), 13),
                   Qt.AlignmentFlag.AlignCenter, str(out)[:18])

    def _draw_array(self, p: QPainter, arr: np.ndarray, rect: QRectF) -> None:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return
        # Detecta formato CHW (canales, alto, ancho) y lo pasa a HWC.
        if a.ndim == 3 and a.shape[2] not in (1, 2, 3, 4) and a.shape[0] <= 128:
            a = np.transpose(a, (1, 2, 0))
        if a.ndim == 3:
            if a.shape[2] == 1:
                a = a[:, :, 0]
            elif a.shape[2] == 2:
                a = a.mean(axis=2)
            else:
                a = a[:, :, :3]
        if a.ndim == 3:
            lo, hi = float(a.min()), float(a.max())
            if hi > 1.5:
                rgb = np.clip(a, 0, 255).astype(np.uint8)
            else:
                rgb = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            from PySide6.QtGui import QImage
            buf = np.ascontiguousarray(rgb)
            qimg = QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            p.drawImage(rect, qimg)
            return
        lo, hi = float(a.min()), float(a.max())
        norm = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        h, w = u8.shape[:2]
        from PySide6.QtGui import QImage
        buf = np.ascontiguousarray(u8)
        qimg = QImage(buf.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        p.drawImage(rect, qimg)

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        hover = -1
        for i, r in enumerate(self._rects):
            if r.adjusted(-2, 0, 2, 0).contains(pos):
                hover = i
                break
        if hover != self._hover:
            self._hover = hover
            self.update()
        if hover >= 0 and preferences.get("tooltips", True):
            row = self._rows[hover]
            tip = (
                f"<b>{row.get('stage', '')}</b><br>{row.get('operation', '')}<br>"
                f"Salida: {row.get('output', '')}<br>"
                f"Parámetros: {row.get('params', '—')}<br>"
                f"<i>{row.get('note', '')}</i>"
            )
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        for i, r in enumerate(self._rects):
            if r.contains(pos):
                self.layerClicked.emit(i)
                break
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = -1
        self.update()
        super().leaveEvent(event)


class MiniArchStrip(QWidget):
    """Tira compacta de capas para la página de entrenamiento."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(84)
        self._rows: list[dict] = []
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if preferences.get("animations", True) and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        if not self.isVisible():
            self._timer.stop()
            return
        if not preferences.get("animations", True):
            self._timer.stop()
            return
        self._phase = (self._phase + 0.05) % 1.0
        self.update()

    def set_rows(self, rows: list) -> None:
        self._rows = rows
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(BG_ALT))
        if not self._rows:
            p.end()
            return
        n = len(self._rows)
        margin = 10
        gap = 6
        bw = max(18.0, (self.width() - 2 * margin - gap * (n - 1)) / n)
        y = 12
        h = self.height() - 26
        for i, row in enumerate(self._rows):
            x = margin + i * (bw + gap)
            color = QColor(row.get("color", ACCENT))
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 26))
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 150), 1))
            p.drawRoundedRect(QRectF(x, y, bw, h), 6, 6)
            p.setPen(QColor(TEXT))
            f = QFont("Segoe UI", 7)
            f.setWeight(QFont.Weight.DemiBold)
            p.setFont(f)
            p.drawText(QRectF(x + 2, y + 3, bw - 4, 13), Qt.AlignmentFlag.AlignCenter,
                       str(row.get("stage", ""))[:9])
            p.setPen(QColor(MUTED))
            p.setFont(QFont("Segoe UI", 6.5))
            out = row.get("output", "")
            p.drawText(QRectF(x + 2, y + 17, bw - 4, 12), Qt.AlignmentFlag.AlignCenter, str(out)[:11])
            p.drawText(QRectF(x + 2, y + h - 14, bw - 4, 12), Qt.AlignmentFlag.AlignCenter,
                       f"{row.get('params', '—')}")
            if i < n - 1:
                t = (self._phase + i / max(1, n)) % 1.0
                px = x + bw + gap * t
                p.setBrush(qcolor(ACCENT, 190))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(px, y + h / 2), 2.0, 2.0)
        p.end()
