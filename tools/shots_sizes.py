"""Capturas de la interfaz en dos resoluciones para revisar el layout responsivo."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme

OUT = ROOT / "tools" / "shots_native"
OUT.mkdir(exist_ok=True)
app = QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.show()
state = {"tries": 0, "jobs": []}


def poll():
    state["tries"] += 1
    if win.dataset_info is None and state["tries"] < 100:
        QTimer.singleShot(400, poll)
        return
    run_sizes()


def run_sizes():
    for w, h in ((1280, 800), (1920, 1080)):
        for page in (0, 1, 2, 4):
            state["jobs"].append((w, h, page))
    state["jobs"].append((1280, 800, 3))
    state["jobs"].append((1920, 1080, 3))
    next_job()


def next_job():
    if not state["jobs"]:
        win.close()
        app.exit(0)
        return
    w, h, page = state["jobs"].pop(0)
    win.resize(w, h)
    win.goto_page(page)
    app.processEvents()
    QTimer.singleShot(1400, lambda: capture(w, h, page))


def capture(w, h, page):
    app.processEvents()
    pm = win.grab()
    path = OUT / f"size_{w}x{h}_page_{page}.png"
    pm.save(str(path))
    print("guardado", path.name, pm.size())
    QTimer.singleShot(250, next_job)


QTimer.singleShot(500, poll)
sys.exit(app.exec())
