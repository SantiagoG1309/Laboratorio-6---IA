"""Tabla compacta para visualizar matrices numéricas con resaltado."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem, QWidget

from ui.theme import ACCENT, BG_ALT, BORDER, BORDER_SOFT, MUTED, MUTED_2, TEXT


class MatrixTable(QTableWidget):
    """Matriz compacta de solo lectura con resaltado de ventanas."""

    cellPicked = Signal(int, int)

    def __init__(self, rows: int = 10, cols: int = 10, editable: bool = False, parent: QWidget | None = None):
        super().__init__(rows, cols, parent)
        self._editable = editable
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            if editable
            else QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.setFixedSize(cols * 31 + 4, rows * 25 + 4)
        self.setStyleSheet(
            f"QTableWidget {{ background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 8px;"
            f"gridline-color: {BORDER_SOFT}; }}"
            "QTableWidget::item { padding: 0px; }"
        )
        self.cellClicked.connect(self.cellPicked.emit)

    def set_matrix(self, values, highlight: tuple[int, int, int, int] | None = None, color: str = ACCENT) -> None:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows, cols = arr.shape
        self.blockSignals(True)
        if rows != self.rowCount() or cols != self.columnCount():
            self.setRowCount(rows)
            self.setColumnCount(cols)
            self.setFixedSize(cols * 31 + 4, rows * 25 + 4)
        hi = QColor(color)
        for r in range(rows):
            for c in range(cols):
                value = arr[r, c]
                text = f"{value:.0f}" if abs(float(value) - round(float(value))) < 1e-6 else f"{float(value):.2f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFont(QFont("Cascadia Code", 8))
                inside = bool(
                    highlight
                    and highlight[0] <= r < highlight[0] + highlight[2]
                    and highlight[1] <= c < highlight[1] + highlight[3]
                )
                if inside:
                    item.setBackground(QColor(hi.red(), hi.green(), hi.blue(), 70))
                    item.setForeground(QColor(TEXT))
                elif float(value) == 0.0:
                    item.setForeground(QColor(MUTED_2))
                else:
                    item.setForeground(QColor(MUTED))
                self.setItem(r, c, item)
        for c in range(cols):
            self.setColumnWidth(c, 31)
        for r in range(rows):
            self.setRowHeight(r, 25)
        self.blockSignals(False)

    def matrix(self) -> np.ndarray:
        rows, cols = self.rowCount(), self.columnCount()
        out = np.zeros((rows, cols), np.float32)
        for r in range(rows):
            for c in range(cols):
                item = self.item(r, c)
                try:
                    out[r, c] = float(item.text()) if item else 0.0
                except ValueError:
                    out[r, c] = 0.0
        return out
