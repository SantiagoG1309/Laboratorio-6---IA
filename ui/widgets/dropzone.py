"""Zona de arrastrar y soltar archivos o carpetas."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from core.config import IMAGE_EXTENSIONS
from ui import icons
from ui.theme import ACCENT, BORDER, MUTED, SURFACE_2


def collect_images(paths: list[str]) -> list[str]:
    out: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    out.append(str(f))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(str(p))
    return out


class DropZone(QFrame):
    pathsDropped = Signal(list)
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(170)
        self._active = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon = QLabel()
        self.icon.setPixmap(icons.pixmap("upload", ACCENT, 38))
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon)
        self.title = QLabel("Arrastra imágenes o una carpeta aquí")
        self.title.setProperty("h3", True)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        self.subtitle = QLabel("o haz clic para seleccionar archivos · JPG, PNG, BMP, WEBP")
        self.subtitle.setProperty("muted", True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.subtitle)
        self._refresh()

    def _refresh(self) -> None:
        color = ACCENT if self._active else BORDER
        bg = "rgba(34,211,238,0.07)" if self._active else SURFACE_2
        self.setStyleSheet(
            f"QFrame {{ background: {bg}; border: 1.5px dashed {color}; border-radius: 14px; }}"
            f"QLabel {{ border: none; background: transparent; color: {MUTED}; }}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._active = True
            self._refresh()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._active = False
        self._refresh()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._active = False
        self._refresh()
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        images = collect_images(paths)
        if images:
            self.pathsDropped.emit(images)
        event.acceptProposedAction()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)
