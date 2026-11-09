"""Visores de imagen con escalado suave y diálogo de ampliación."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui import icons
from ui.theme import ACCENT, BG_ALT, BORDER, MUTED, MUTED_2, SURFACE, SURFACE_2, TEXT


def numpy_to_qimage(img: np.ndarray) -> QImage:
    if img is None:
        return QImage()
    arr = np.ascontiguousarray(img)
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        h, w, _ = arr.shape
        return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    return QImage()


def numpy_to_pixmap(img: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(img))


class ImageView(QWidget):
    doubleClicked = Signal()

    def __init__(
        self,
        placeholder: str = "Sin imagen",
        icon_name: str = "image",
        parent: QWidget | None = None,
        min_height: int = 150,
        rounded: int = 10,
    ):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self._icon_name = icon_name
        self._rounded = rounded
        self.setMinimumHeight(min_height)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_image(self, img: np.ndarray | QPixmap | None) -> None:
        if isinstance(img, QPixmap):
            self._pixmap = img
        elif isinstance(img, np.ndarray) and img.size:
            self._pixmap = numpy_to_pixmap(img)
        else:
            self._pixmap = None
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is not None:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, self._rounded, self._rounded)
        p.fillPath(path, QColor(BG_ALT))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                int(rect.width()), int(rect.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = rect.x() + (rect.width() - scaled.width()) / 2
            y = rect.y() + (rect.height() - scaled.height()) / 2
            p.save()
            p.setClipPath(path)
            p.drawPixmap(int(x), int(y), scaled)
            p.restore()
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawPath(path)
        else:
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawPath(path)
            size = 34
            pm = icons.pixmap(self._icon_name, BORDER, size)
            p.drawPixmap(int(rect.center().x() - size / 2), int(rect.center().y() - size - 2), pm)
            p.setPen(QColor(MUTED_2))
            p.drawText(rect.adjusted(0, 24, 0, 0), Qt.AlignmentFlag.AlignCenter, self._placeholder)
        p.end()


class ImageDialog(QDialog):
    def __init__(self, pixmap: QPixmap, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        head = QHBoxLayout()
        col = QVBoxLayout()
        t = QLabel(title)
        t.setProperty("h2", True)
        col.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setProperty("muted", True)
            col.addWidget(s)
        head.addLayout(col, 1)
        close = QPushButton("Cerrar")
        close.setProperty("variant", "ghost")
        close.clicked.connect(self.accept)
        head.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)
        view = QLabel()
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setStyleSheet(f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;")
        view.setPixmap(pixmap.scaled(940, 640, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(view, 1)


def open_image_dialog(parent: QWidget, img: np.ndarray | QPixmap, title: str, subtitle: str = "") -> None:
    pm = img if isinstance(img, QPixmap) else numpy_to_pixmap(img)
    if pm.isNull():
        return
    dlg = ImageDialog(pm, title, subtitle, parent)
    dlg.exec()
