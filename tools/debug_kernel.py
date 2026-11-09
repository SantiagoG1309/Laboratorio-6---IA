"""Debug temporal: estado y aspecto del editor de kernels."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme

out = ROOT / "tools" / "shots_native"
out.mkdir(exist_ok=True)
app = QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1480, 920)
win.show()
state = {"tries": 0}


def poll():
    state["tries"] += 1
    if win.dataset_info is None and state["tries"] < 100:
        QTimer.singleShot(300, poll)
        return
    QTimer.singleShot(900, run)


def run():
    win.goto_page(1)
    app.processEvents()
    ctl = win.explorer.controls
    combo = ctl.kernel_combo
    idx = combo.findData("custom")
    combo.setCurrentIndex(idx)
    app.processEvents()
    ctl.kernel_editor.set_values([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    app.processEvents()
    print("editable:", [b.isEnabled() for b in ctl.kernel_editor._boxes])
    print("valores:", ctl.kernel_editor.values())
    print("config kernel:", ctl.config().kernel)
    print("box size:", ctl.kernel_editor._boxes[0].size(), "text:", ctl.kernel_editor._boxes[0].text())
    ctl.kernel_editor.grab().scaled(620, 200).save(str(out / "kernel_editor.png"))
    win.grab().save(str(out / "pipeline_custom.png"))
    print("guardado")
    win.close()
    app.exit(0)


QTimer.singleShot(400, poll)
sys.exit(app.exec())
