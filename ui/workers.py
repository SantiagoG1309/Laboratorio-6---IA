"""Tareas en segundo plano: pipeline, escaneo, entrenamiento y exportación."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThread, Signal

from core.config import PipelineConfig, TrainingConfig, THUMBS_PER_CLASS
from core.dataset import DatasetInfo, Sample, scan_dataset, split_dataset
from core.exporter import export_processed_dataset
from core.pipeline import analyze_image
from core.training import TrainingResult, predict_paths, train_model
from utils.thumbnails import ensure_thumbs


class PipelineSignals(QObject):
    result = Signal(object)
    error = Signal(str)


class PipelineWorker(QRunnable):
    def __init__(self, path: str, cfg: PipelineConfig, token: int = 0):
        super().__init__()
        self.path = path
        self.cfg = cfg
        self.token = token
        self.signals = PipelineSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            res = analyze_image(self.path, self.cfg)
            self.signals.result.emit((self.token, res))
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(f"{exc}\n{traceback.format_exc(limit=2)}")
            except RuntimeError:
                pass


class ScanSignals(QObject):
    done = Signal(object, dict)
    error = Signal(str)


class ScanWorker(QRunnable):
    """Escanea el dataset y genera miniaturas en caché."""

    def __init__(self, root: str | Path):
        super().__init__()
        self.root = str(root)
        self.signals = ScanSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            info = scan_dataset(self.root)
            # Solo se generan miniaturas para una muestra por clase: con miles de
            # imágenes el escaneo seguiría siendo rápido.
            from collections import defaultdict

            from ui import preferences

            limit = int(preferences.get("thumbs", THUMBS_PER_CLASS))
            buckets: dict[str, list[str]] = defaultdict(list)
            for s in info.samples:
                if len(buckets[s.class_name]) < limit:
                    buckets[s.class_name].append(str(s.path))
            paths = [p for bucket in buckets.values() for p in bucket]
            thumbs = ensure_thumbs(paths, 240, 180)
            self.signals.done.emit(info, thumbs)
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                pass


class TrainingThread(QThread):
    log = Signal(str)
    epoch = Signal(dict)
    batch = Signal(int, int, int, float)
    phase = Signal(dict)
    done = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        train: list[Sample],
        val: list[Sample],
        test: list[Sample],
        cfg: TrainingConfig,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.train = train
        self.val = val
        self.test = test
        self.cfg = cfg
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: D401
        try:
            result: TrainingResult = train_model(
                self.train,
                self.val,
                self.test,
                self.cfg,
                on_epoch=lambda info: self.epoch.emit(info),
                on_batch=lambda e, b, t, loss: self.batch.emit(e, b, t, loss),
                on_log=lambda msg: self.log.emit(msg),
                should_stop=self._stop.is_set,
                on_phase=lambda info: self.phase.emit(info),
            )
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


class ExportThread(QThread):
    progress = Signal(int, int, str)
    log = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, samples: list[Sample], cfg: PipelineConfig, output: str, parent: QObject | None = None):
        super().__init__(parent)
        self.samples = samples
        self.cfg = cfg
        self.output = output
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: D401
        try:
            path = export_processed_dataset(
                self.samples,
                self.cfg,
                self.output,
                progress=lambda i, n, name: self.progress.emit(i, n, name),
                log=lambda msg: self.log.emit(msg),
                should_stop=self._stop.is_set,
            )
            self.done.emit(str(path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


class PredictSignals(QObject):
    done = Signal(object, object)
    error = Signal(str)


class DatasetPrepSignals(QObject):
    done = Signal(int)
    error = Signal(str)

class DatasetPrepWorker(QRunnable):
    """Copia el dataset incluido la primera vez, sin congelar la interfaz."""

    def __init__(self):
        super().__init__()
        self.signals = DatasetPrepSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            from utils import paths

            copied = paths.copy_bundled_dataset() if paths.dataset_needs_copy() else 0
            self.signals.done.emit(copied)
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                pass


class PredictWorker(QRunnable):
    def __init__(self, model, paths: list[str], cfg: TrainingConfig):
        super().__init__()
        self.model = model
        self.paths = paths
        self.cfg = cfg
        self.signals = PredictSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            probs, pred = predict_paths(self.model, self.paths, self.cfg)
            self.signals.done.emit(probs, pred)
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                pass


class ThumbSignals(QObject):
    done = Signal(dict)


class ThumbWorker(QRunnable):
    """Genera miniaturas en segundo plano para no bloquear la interfaz."""

    def __init__(self, paths: list[str], width: int = 300, height: int = 220):
        super().__init__()
        self.paths = list(paths)
        self.width = int(width)
        self.height = int(height)
        self.signals = ThumbSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        try:
            thumbs = ensure_thumbs(self.paths, self.width, self.height)
            self.signals.done.emit(thumbs)
        except RuntimeError:
            pass
        except Exception:  # noqa: BLE001
            try:
                self.signals.done.emit({})
            except RuntimeError:
                pass


def make_splits(info: DatasetInfo, cfg: TrainingConfig) -> dict[str, list[Sample]]:
    return split_dataset(
        info.samples,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        seed=cfg.seed,
        stratified=True,
    )
