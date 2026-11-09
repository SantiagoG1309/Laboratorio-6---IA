"""Capturas nativas (usa la plataforma Windows real) para el README."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme

out = ROOT / "docs" / "capturas"
out.mkdir(parents=True, exist_ok=True)
NAMES = ["inicio", "pipeline", "arquitectura", "entrenamiento", "resultados"]
app = QApplication(sys.argv)
apply_theme(app)
win = MainWindow()
win.resize(1480, 920)
win.show()
state = {"tries": 0}


def poll():
    state["tries"] += 1
    if (win.dataset_info is None or win.loaded_cfg is None) and state["tries"] < 100:
        QTimer.singleShot(400, poll)
        return
    # Alinea la vista de arquitectura con el modelo incluido (sin guardar nada).
    if win.loaded_cfg is not None:
        win.training_cfg = win.loaded_cfg
        win.training._apply_config(win.loaded_cfg)
        win.architecture.refresh_from_training()
    QTimer.singleShot(1500, lambda: show_page(0))


def show_page(i: int):
    if i >= 5:
        win.close()
        app.exit(0)
        return
    win.goto_page(i)
    app.processEvents()
    delay = 2600 if i == 1 else (3500 if i == 4 else 900)
    QTimer.singleShot(delay, lambda idx=i: capture(idx))


def capture(i: int):
    app.processEvents()
    pm = win.grab()
    p = out / f"{NAMES[i]}.png"
    pm.save(str(p))
    print("guardado", p, pm.size())
    QTimer.singleShot(250, lambda: show_page(i + 1))


QTimer.singleShot(500, poll)
sys.exit(app.exec())
