"""Componentes visuales reutilizables: tarjetas, badges, switches y estados vacíos."""
from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui import icons
from ui import preferences
from ui.glossary import GlossaryLabel
from ui.theme import (
    ACCENT,
    BORDER,
    KNOB_OFF,
    KNOB_ON,
    MUTED,
    MUTED_2,
    ON_ACCENT,
    SURFACE_2,
    TEXT,
    TOGGLE_OFF,
    mix,
    rgba,
)


class Card(QFrame):
    def __init__(self, parent: QWidget | None = None, padding: int = 18, spacing: int = 10):
        super().__init__(parent)
        self.setProperty("card", True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(spacing)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)


class SoftPanel(QFrame):
    def __init__(self, parent: QWidget | None = None, padding: int = 12, spacing: int = 8):
        super().__init__(parent)
        self.setProperty("soft", True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(spacing)

    def body(self) -> QVBoxLayout:
        return self._layout


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None, tip: str = ""):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = GlossaryLabel(title)
        self.title.setProperty("h2", True)
        col.addWidget(self.title)
        self.subtitle = GlossaryLabel(subtitle)
        self.subtitle.setProperty("muted", True)
        col.addWidget(self.subtitle)
        row.addLayout(col, 1)
        self.subtitle.setVisible(bool(subtitle))
        self._right = QHBoxLayout()
        self._right.setSpacing(8)
        row.addLayout(self._right)
        hint = tip or subtitle
        if hint:
            self.setToolTip(hint)
            self.title.setToolTip(hint)
            self.subtitle.setToolTip(hint)

    def add_right(self, widget: QWidget) -> None:
        self._right.addWidget(widget)


class StatCard(Card):
    def __init__(self, icon_name: str, label: str, value: str = "—", accent: str = ACCENT, parent: QWidget | None = None):
        super().__init__(parent, padding=16, spacing=6)
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._icon_name = icon_name
        row = QHBoxLayout()
        row.setSpacing(12)
        self.icon_box = QLabel()
        self.icon_box.setFixedSize(42, 42)
        self.icon_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_box.setPixmap(icons.pixmap(icon_name, accent, 22))
        self.icon_box.setStyleSheet(
            f"background: {rgba(accent, 0.10)}; border: 1px solid {rgba(accent, 0.28)}; border-radius: 12px;"
        )
        row.addWidget(self.icon_box)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.value = QLabel(value)
        f = QFont()
        f.setPointSize(19)
        f.setWeight(QFont.Weight.Bold)
        self.value.setFont(f)
        self.value.setStyleSheet(f"color: {TEXT};")
        self.label = GlossaryLabel(label)
        self.label.setProperty("muted", True)
        self.label.setStyleSheet("font-size: 11px;")
        col.addWidget(self.value)
        col.addWidget(self.label)
        row.addLayout(col, 1)
        self.body().addLayout(row)
        self.accent = accent

    def set_value(self, text: str) -> None:
        self.value.setText(text)

    def set_accent(self, color: str) -> None:
        self.icon_box.setPixmap(icons.pixmap(self._icon_name if hasattr(self, "_icon_name") else "target", color, 22))


class Badge(QLabel):
    def __init__(self, text: str, color: str = ACCENT, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background: rgba(255,255,255,0.035); color: {color}; border: 1px solid {rgba(color, 0.33)};"
            "border-radius: 10px; padding: 2px 9px; font-size: 11px; font-weight: 600;"
        )
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


class Pill(QLabel):
    def __init__(self, text: str = "", icon_name: str | None = None, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setProperty("muted", True)
        self.setStyleSheet(
            f"background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 11px;"
            f"padding: 4px 10px; color: {MUTED}; font-size: 11px;"
        )
        if icon_name:
            self.setText(("  " if text else "") + text)


class Divider(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {BORDER}; border: none;")


class ElidedLabel(QLabel):
    """Etiqueta de una línea que recorta con «…» cuando falta espacio."""

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle,
    ):
        super().__init__(parent)
        self._full_text = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(90)
        super().setText(text)
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full_text = text
        self.setToolTip(text)
        self._apply_elide()

    def text(self) -> str:  # noqa: N802
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self) -> None:
        width = max(30, self.width() - 2)
        elided = self.fontMetrics().elidedText(self._full_text, self._mode, width)
        super().setText(elided)


class ToggleSwitch(QAbstractButton):
    """Interruptor animado tipo iOS."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(46, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._offset = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        """Sincroniza el dibujo aunque las señales estén bloqueadas."""
        super().setChecked(checked)
        if self.signalsBlocked():
            self._anim.stop()
            self.set_offset(1.0 if checked else 0.0)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        target = 1.0 if checked else 0.0
        if not preferences.get("animations", True):
            self.set_offset(target)
            return
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(target)
        self._anim.start()

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = Property(float, get_offset, set_offset)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        track = QRectF(0, 1, self.width(), self.height() - 2)
        enabled = self.isEnabled()
        checked = self.isChecked()
        if not enabled:
            if checked:
                p.setBrush(QColor(mix(SURFACE_2, ACCENT, 0.55)))
                p.setPen(QPen(QColor(rgba(ACCENT, 0.55)), 1))
            else:
                p.setBrush(QColor(SURFACE_2))
                p.setPen(QPen(QColor(BORDER), 1))
        elif checked:
            p.setBrush(QColor(ACCENT))
            p.setPen(Qt.PenStyle.NoPen)
        else:
            p.setBrush(QColor(TOGGLE_OFF))
            p.setPen(QPen(QColor(BORDER), 1))
        p.drawRoundedRect(track, track.height() / 2, track.height() / 2)
        knob_d = self.height() - 8
        x = 4 + self._offset * (self.width() - knob_d - 8)
        p.setPen(Qt.PenStyle.NoPen)
        if not enabled:
            p.setBrush(QColor(mix(KNOB_OFF, KNOB_ON, 0.75) if checked else KNOB_OFF))
        else:
            p.setBrush(QColor(KNOB_ON if checked else KNOB_OFF))
        p.drawEllipse(QRectF(x, 4, knob_d, knob_d))
        p.end()


class EmptyState(QWidget):
    def __init__(
        self,
        icon_name: str,
        title: str,
        description: str,
        parent: QWidget | None = None,
        accent: str = ACCENT,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 30, 24, 30)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setPixmap(icons.pixmap(icon_name, accent, 56))
        layout.addWidget(self.icon_label)
        self.title = GlossaryLabel(title)
        self.title.setProperty("h2", True)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        self.description = QLabel(description)
        self.description.setProperty("muted", True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setWordWrap(True)
        self.description.setMaximumWidth(520)
        self.description.setMinimumWidth(0)
        layout.addWidget(self.description, 0, Qt.AlignmentFlag.AlignHCenter)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(10)
        self.buttons.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(self.buttons)

    def add_button(self, button: QPushButton) -> None:
        self.buttons.addWidget(button)


def shadow(widget: QWidget, blur: int = 30, alpha: int = 70) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, 8)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)


class ColorDot(QLabel):
    def __init__(self, color: str, size: int = 9, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setStyleSheet(f"background: {color}; border-radius: {size // 2}px;")


class KeyValue(QWidget):
    def __init__(self, key: str, value: str, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.key = QLabel(key)
        self.key.setProperty("muted", True)
        self.value = QLabel(value)
        self.value.setStyleSheet("font-weight: 600;")
        row.addWidget(self.key)
        row.addStretch(1)
        row.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)
