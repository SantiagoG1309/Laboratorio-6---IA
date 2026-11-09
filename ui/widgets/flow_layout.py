"""Layout de flujo responsivo para tarjetas y galerías."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Distribuye widgets en filas y los estira para ocupar el ancho disponible.

    ``min_item_width`` fija el ancho mínimo de celda usado para decidir cuántas
    columnas caben. Si es 0 se deduce del mínimo de los widgets.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        margin: int = 0,
        h_spacing: int = 12,
        v_spacing: int = 12,
        min_item_width: int = 0,
    ):
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing
        self._min_item_width = min_item_width
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, max(1, width), 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        if self._min_item_width:
            size.setWidth(max(size.width(), int(self._min_item_width)))
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _columns(self, width: int) -> tuple[int, int]:
        if not self._items:
            return 1, max(1, width)
        base = self._min_item_width
        if base <= 0:
            base = max(item.minimumSize().width() for item in self._items)
        base = max(80, base)
        cols = max(1, (width + self._h) // (base + self._h))
        cell = max(1, (width - (cols - 1) * self._h) // cols)
        return int(cols), int(cell)

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        width = max(1, effective.width())
        cols, cell = self._columns(width)

        x = effective.x()
        y = effective.y()
        line_height = 0
        for index, item in enumerate(self._items):
            if index and index % cols == 0:
                x = effective.x()
                y += line_height + self._v
                line_height = 0
            hint = item.sizeHint()
            w = min(cell, item.maximumSize().width())
            w = max(w, item.minimumSize().width())
            if not test_only:
                item.setGeometry(QRect(QPoint(int(x), int(y)), QSize(int(w), hint.height())))
            x += w + self._h
            line_height = max(line_height, hint.height())
        return int(y + line_height - rect.y() + margins.bottom())


class ResponsiveGrid(QWidget):
    """Contenedor con FlowLayout que se adapta al ancho disponible."""

    def __init__(self, spacing: int = 12, min_item_width: int = 0, parent: QWidget | None = None):
        super().__init__(parent)
        self._flow = FlowLayout(
            self,
            margin=0,
            h_spacing=spacing,
            v_spacing=spacing,
            min_item_width=min_item_width,
        )
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._flow.heightForWidth(max(1, width))

    def sizeHint(self) -> QSize:  # noqa: N802
        width = max(1, self.width())
        return QSize(super().sizeHint().width(), self._flow.heightForWidth(width))

    def add(self, widget: QWidget) -> None:
        self._flow.addWidget(widget)

    def clear(self) -> None:
        while self._flow.count():
            item = self._flow.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
