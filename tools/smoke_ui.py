"""Prueba de arranque sin pantalla (offscreen) para detectar errores de UI."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme

app = QApplication(sys.argv)
apply_theme(app)
errors: list[str] = []


def hook(exc_type, exc, tb):
    errors.append("".join(traceback.format_exception(exc_type, exc, tb)))
    traceback.print_exception(exc_type, exc, tb)


sys.excepthook = hook
win = MainWindow()
win.show()

state = {"tries": 0}


def poll():
    state["tries"] += 1
    if state["tries"] % 5 == 0:
        print("poll", state["tries"], "info", win.dataset_info, "path", win.dataset_path, flush=True)
    if win.dataset_info is not None:
        print("dataset:", win.dataset_info.total, win.dataset_info.counts)
        print("splits:", {k: len(v) for k, v in (win.splits or {}).items()})
        for i in range(len(win.TITLES)):
            win.goto_page(i)
            app.processEvents()
        print("pages ok", len(win.TITLES))
        print("lab steps:", len(win.architecture._lab_steps))
        print("diagram rows:", len(win.architecture._rows))
        print("theme:", win.settings_page.theme_combo.currentData(), win.settings_page.accent_combo.currentData())
        QTimer.singleShot(800, finish)
        return
    if state["tries"] > 60:
        print("TIMEOUT esperando escaneo")
        finish()
        return
    QTimer.singleShot(500, poll)


def finish():
    win.close()
    app.processEvents()
    if errors:
        print("ERRORES:", len(errors))
        for e in errors[:5]:
            print(e)
        app.exit(1)
    else:
        print("SIN ERRORES")
        app.exit(0)


QTimer.singleShot(600, poll)
sys.exit(app.exec())
