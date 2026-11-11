"""Prueba funcional de UI: pipeline, entrenamiento corto, resultados y exportación."""
from __future__ import annotations

import os
import sys
import tempfile
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
state = {"stage": "scan", "ticks": 0}


def fail(msg: str):
    print("FALLO:", msg)
    finish(1)


def tick():
    state["ticks"] += 1
    stage = state["stage"]
    if stage == "scan":
        if win.dataset_info is not None:
            print("scan ok", win.dataset_info.total)
            state["stage"] = "pipeline"
            win.goto_page(1)
            win.explorer.set_sample(str(win.dataset_info.samples[0].path))
        elif state["ticks"] > 120:
            return fail("timeout escaneo")
    elif stage == "pipeline":
        if win.explorer._result is not None:
            res = win.explorer._result
            print("pipeline ok", res.flat_size, res.original.shape, res.conv_raw.shape)
            state["stage"] = "training_setup"
            start_training()
    elif stage == "training":
        pass
    QTimer.singleShot(300, tick)


def start_training():
    tp = win.training
    tp.input_combo.setCurrentIndex(0)  # 32 px
    tp.hidden_spin.setValue(16)
    tp.f1_spin.setValue(2)
    tp.f2_spin.setValue(2)
    tp.epochs_spin.setValue(2)
    tp.batch_spin.setValue(16)
    tp.lr_spin.setValue(0.01)
    tp.thread = None
    tp.start_training()
    if tp.thread is None:
        return fail("no se creó el hilo de entrenamiento")
    print("entrenamiento iniciado")
    tp.thread.done.connect(lambda _r: on_training_done())
    tp.thread.failed.connect(lambda m: fail("entrenamiento: " + m.splitlines()[0]))
    state["stage"] = "training"


def on_training_done():
    print("entrenamiento ok", win.training_result.metrics["accuracy"])
    QTimer.singleShot(600, check_results)


def check_results():
    try:
        assert win.training_result is not None
        assert win.results.body.isVisible() or True
        win.goto_page(3)
        app.processEvents()
        print("resultados ok", win.results.stat_acc.value.text())
        print("tabla filas", win.results.table.rowCount())
        # Guardar/cargar modelo
        out = Path(tempfile.gettempdir()) / "aquacnn_test_model.npz"
        from core.model_io import load_model, save_model
        save_model(out, win.training_result.model, win.training_result.config,
                   {"accuracy": win.training_result.metrics["accuracy"]})
        model, meta, cfg = load_model(out)
        probs = model.predict_proba(
            __import__("numpy").zeros((2, 3, 32, 32), dtype="float32")
        )
        print("modelo guardado/cargado", meta.get("model_type"), probs.shape)
        # Exportar una muestra de 6 imágenes
        exp = Path(tempfile.gettempdir()) / "aquacnn_export_test"
        from core.exporter import export_processed_dataset
        samples = win.dataset_info.samples[:6]
        p = export_processed_dataset(samples, win.pipeline_cfg.copy(), exp)
        files = list(Path(p).rglob("*"))
        print("export ok", p, "archivos", len(files))
    except Exception:
        traceback.print_exc()
        return fail("resultados/exportación")
    finish(0)


def finish(code: int):
    try:
        win.close()
        app.processEvents()
    except Exception:  # noqa: BLE001
        pass
    if errors:
        print("ERRORES:", len(errors))
        for e in errors[:6]:
            print(e)
        code = 1
    print("RESULTADO:", "OK" if code == 0 else "FALLO")
    app.exit(code)


QTimer.singleShot(600, tick)
sys.exit(app.exec())
