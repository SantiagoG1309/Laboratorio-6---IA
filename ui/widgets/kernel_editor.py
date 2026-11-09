"""Editor 3×3 de kernels de convolución con valores siempre visibles."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QSpinBox, QWidget

from ui.theme import ACCENT, DANGER, MUTED, MUTED_2


class KernelEditor(QWidget):
    """Matriz 3×3 editable; los valores se colorean según su signo.

    Se usan QSpinBox con los botones nativos visibles porque, con el QSS del
    tema, ocultarlos dejaba el campo de texto en 1 px y los números no se veían.
    """

    valueChanged = Signal(list)

    def __init__(self, values: list[list[float]] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(5)
        self._boxes: list[QSpinBox] = []
        for r in range(3):
            for c in range(3):
                box = QSpinBox()
                box.setRange(-10, 10)
                box.setSingleStep(1)
                box.setAlignment(Qt.AlignmentFlag.AlignCenter)
                box.setFixedSize(64, 38)
                box.setToolTip("Valor del kernel (−10 a 10)")
                box.valueChanged.connect(self._emit)
                box.valueChanged.connect(lambda _v, b=box: self._style_box(b))
                grid.addWidget(box, r, c)
                self._boxes.append(box)
        self.set_values(values or [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    def _emit(self) -> None:
        self.valueChanged.emit(self.values())

    def _style_box(self, box: QSpinBox) -> None:
        value = box.value()
        color = DANGER if value < 0 else (ACCENT if value > 0 else MUTED)
        box.setStyleSheet(
            f"QSpinBox {{ color: {color}; font-weight: 600; }}"
            f"QSpinBox:disabled {{ color: {MUTED_2}; }}"
        )

    def values(self) -> list[list[float]]:
        return [[float(self._boxes[r * 3 + c].value()) for c in range(3)] for r in range(3)]

    def set_values(self, values: list[list[float]]) -> None:
        for r in range(3):
            for c in range(3):
                box = self._boxes[r * 3 + c]
                box.blockSignals(True)
                box.setValue(int(round(values[r][c])))
                box.blockSignals(False)
                self._style_box(box)

    def set_editable(self, editable: bool) -> None:
        for box in self._boxes:
            box.setEnabled(editable)
