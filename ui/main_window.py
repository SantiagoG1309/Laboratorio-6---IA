"""Ventana principal: navegación, estado del dataset y orquestación de páginas."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QSettings, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import utils.paths as paths
from core.config import CLASSES, CLASS_LABELS, PipelineConfig, TrainingConfig, APP_TITLE, MAX_DATASET_IMAGES
from core.dataset import DatasetInfo, ensure_structure, import_images, scan_dataset
from core.metrics import evaluate_predictions
from core.model_io import load_model
from core.training import TrainingResult
from ui import icons, preferences
from ui.pages.about import AboutPage
from ui.pages.architecture import ArchitecturePage
from ui.pages.dashboard import DashboardPage
from ui.pages.explorer import ExplorerPage
from ui.pages.results import ResultsPage
from ui.pages.settings import SettingsPage
from ui.pages.training import TrainingPage
from ui.theme import (
    ACCENT,
    BG_ALT,
    BORDER,
    DANGER,
    MUTED,
    MUTED_2,
    ON_ACCENT,
    SUCCESS,
    SURFACE_2,
    TEXT,
    WARNING,
    rgba,
)
from ui.widgets.sidebar import Sidebar
from ui.workers import DatasetPrepWorker, PredictWorker, ScanWorker, make_splits


class Toast(QFrame):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setVisible(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 16, 10)
        lay.setSpacing(9)
        self.icon = QLabel()
        self.icon.setFixedSize(18, 18)
        lay.addWidget(self.icon)
        self.label = QLabel("")
        self.label.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(460)
        lay.addWidget(self.label)
        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        self._anim = QPropertyAnimation(self.effect, b"opacity", self)
        self._anim.setDuration(180)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)

    def show_message(self, text: str, kind: str = "info") -> None:
        colors = {"info": ACCENT, "success": SUCCESS, "warning": WARNING, "error": DANGER}
        color = colors.get(kind, ACCENT)
        icon_name = {"info": "info", "success": "check", "warning": "info", "error": "x"}.get(kind, "info")
        self.icon.setPixmap(icons.pixmap(icon_name, color, 17))
        self.label.setText(text)
        self.setStyleSheet(
            f"QFrame {{ background: {SURFACE_2}; border: 1px solid {rgba(color, 0.4)}; border-left: 3px solid {color};"
            "border-radius: 10px; }}"
            "QLabel { background: transparent; border: none; }"
        )
        self.adjustSize()
        self._reposition()
        self.setVisible(True)
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()
        self._timer.start(3800)

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._hide_once)
        self._anim.start()

    def _hide_once(self) -> None:
        try:
            self._anim.finished.disconnect(self._hide_once)
        except Exception:  # noqa: BLE001
            pass
        self.setVisible(False)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        x = parent.width() - self.width() - 24
        y = parent.height() - self.height() - 24
        self.move(max(10, x), max(10, y))


class MainWindow(QMainWindow):
    pipeline_changed = Signal()

    TITLES = [
        ("Inicio", "Estado del dataset y flujo del laboratorio"),
        ("Pipeline visual", "Explora cada etapa del tratamiento de imagen en vivo"),
        ("Arquitectura NN", "Diagrama de neuronas, tablas vivas y laboratorio numérico"),
        ("Entrenamiento", "Configura y monitorea la red neuronal en segundo plano"),
        ("Resultados", "Métricas, matriz de confusión y predicciones"),
        ("Acerca del laboratorio", "Teoría, fórmulas y créditos"),
        ("Configuración", "Apariencia, animaciones, tooltips y preferencias"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1024, 620)
        self.resize(1480, 920)
        self.setWindowIcon(icons.app_icon())
        self.settings = QSettings("AquaCNN", "Laboratorio6")
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 2)))

        self.dataset_path = self._initial_dataset_path()
        self.pipeline_cfg = self._load_cfg("pipeline", PipelineConfig(), PipelineConfig.from_dict)
        self.training_cfg = self._load_cfg("training", TrainingConfig(), TrainingConfig.from_dict)
        self.models_dir = paths.models_dir()
        self.exports_dir = paths.exports_dir()
        self.dataset_info: DatasetInfo | None = None
        self.dataset_thumbs: dict[str, str] = {}
        self.splits: dict | None = None
        self.training_result: TrainingResult | None = None
        self.loaded_model = None
        self.loaded_meta: dict = {}
        self.loaded_cfg: TrainingConfig | None = None
        self.loaded_path: Path | None = None
        self._default_evaluated = False
        self._scan_token = 0
        self._scan_workers: list[ScanWorker] = []
        self._predict_workers: list[PredictWorker] = []
        self._prep_workers: list[DatasetPrepWorker] = []

        self._build_ui()
        self._shortcuts()
        self.goto_page(int(self.settings.value("last_page", 0)))
        QTimer.singleShot(120, self._load_default_model)
        QTimer.singleShot(150, self._prepare_bundled_dataset)

    # ------------------------------------------------------------------
    def _apply_window_icon(self) -> None:
        """Vuelve a fijar el icono: Windows recrea el HWND al maximizar."""
        icon = icons.app_icon()
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def showEvent(self, event) -> None:  # noqa: N802
        self._apply_window_icon()
        super().showEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == QEvent.Type.WinIdChange:
            self._apply_window_icon()
        super().changeEvent(event)

    # ------------------------------------------------------------------
    def _load_cfg(self, key: str, default, factory):
        raw = self.settings.value(key, "")
        if not raw:
            return default
        try:
            return factory(json.loads(str(raw)))
        except Exception:  # noqa: BLE001
            return default

    def _initial_dataset_path(self) -> Path:
        """Usa la última carpeta válida o el dataset portable del usuario."""
        stored = str(self.settings.value("dataset_path", "") or "")
        if stored:
            candidate = Path(stored)
            if candidate.exists():
                return candidate
        try:
            return paths.ensure_structure(paths.dataset_dir())
        except OSError:
            return paths.ensure_dataset()

    def _prepare_bundled_dataset(self) -> None:
        """Copia el dataset de ejemplo en segundo plano la primera vez."""
        default_dataset = paths.dataset_dir()
        try:
            same = self.dataset_path.resolve() == default_dataset.resolve()
        except OSError:
            same = False
        if not same or not paths.dataset_needs_copy():
            self.load_dataset(self.dataset_path, initial=True)
            return
        self.set_status("Preparando el dataset de ejemplo (solo la primera vez)…")
        worker = DatasetPrepWorker()
        self._prep_workers.append(worker)
        worker.signals.done.connect(self._on_dataset_prepared)
        worker.signals.error.connect(self._on_dataset_prep_failed)
        worker.signals.done.connect(lambda *_: self._release_worker(self._prep_workers, worker))
        worker.signals.error.connect(lambda *_: self._release_worker(self._prep_workers, worker))
        self.thread_pool.start(worker)

    def _on_dataset_prepared(self, copied: int) -> None:
        self.load_dataset(self.dataset_path, initial=True)
        if copied:
            self.toast(f"Dataset de ejemplo listo ({copied} imágenes).", "success")

    def _on_dataset_prep_failed(self, message: str) -> None:
        self.toast(f"No se pudo preparar el dataset de ejemplo: {message}", "error")
        self.load_dataset(self.dataset_path, initial=True)

    def _save_cfg(self) -> None:
        self.settings.setValue("pipeline", json.dumps(self.pipeline_cfg.to_dict()))
        self.settings.setValue("training", json.dumps(self.training_cfg.to_dict()))
        self.settings.setValue("dataset_path", str(self.dataset_path))

    def _load_default_model(self) -> None:
        """Carga el modelo preentrenado incluido para que la app funcione al abrir."""
        try:
            paths.ensure_models()
            chosen = paths.default_model_path()
            if chosen is None:
                return
            model, meta, cfg = load_model(chosen)
        except Exception as exc:  # noqa: BLE001
            self.set_status(f"No se pudo cargar el modelo incluido: {exc}")
            return
        self.loaded_model = model
        self.loaded_meta = meta
        self.loaded_cfg = cfg
        self.loaded_path = chosen
        self.training.set_loaded_model(meta, chosen.name)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.pageChanged.connect(self.goto_page)
        root.addWidget(self.sidebar)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(0)
        root.addWidget(right, 1)

        # Top bar
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(78)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(24, 12, 24, 12)
        tl.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.page_title = QLabel("Inicio")
        self.page_title.setProperty("h2", True)
        self.page_subtitle = QLabel("")
        self.page_subtitle.setProperty("muted", True)
        self.page_subtitle.setStyleSheet("font-size: 11px;")
        titles.addWidget(self.page_title)
        titles.addWidget(self.page_subtitle)
        tl.addLayout(titles, 1)

        self.dataset_chip = QPushButton("0 imágenes")
        self.dataset_chip.setProperty("variant", "ghost")
        self.dataset_chip.setIcon(icons.icon("database", ACCENT, 16))
        self.dataset_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dataset_chip.setToolTip("Abrir carpeta del dataset")
        self.dataset_chip.clicked.connect(self.open_dataset_folder)
        tl.addWidget(self.dataset_chip)

        refresh_btn = QPushButton()
        refresh_btn.setProperty("variant", "icon")
        refresh_btn.setIcon(icons.icon("refresh", MUTED, 18))
        refresh_btn.setFixedSize(36, 36)
        refresh_btn.setToolTip("Volver a escanear el dataset (Ctrl+R)")
        refresh_btn.clicked.connect(self.rescan_dataset)
        tl.addWidget(refresh_btn)

        load_btn = QPushButton("Cargar dataset")
        load_btn.setProperty("variant", "primary")
        load_btn.setIcon(icons.icon("folder", ON_ACCENT, 16))
        load_btn.setToolTip("Elegir otra carpeta que contenga trucha/, mojarra/ y pavon/ (Ctrl+O)")
        load_btn.clicked.connect(self.choose_dataset_folder)
        tl.addWidget(load_btn)
        right_l.addWidget(top)

        # Stack
        self.stack = QStackedWidget()
        self.dashboard = DashboardPage(self)
        self.explorer = ExplorerPage(self)
        self.architecture = ArchitecturePage(self)
        self.training = TrainingPage(self)
        self.results = ResultsPage(self)
        self.about = AboutPage()
        self.settings_page = SettingsPage(self)
        for page in (
            self.dashboard, self.explorer, self.architecture, self.training,
            self.results, self.about, self.settings_page,
        ):
            self.stack.addWidget(page)
        right_l.addWidget(self.stack, 1)
        self.training.configChanged.connect(self._on_training_config)
        self.pipeline_changed.connect(self.architecture.refresh_from_training)
        try:
            self.explorer.controls.auto_toggle._switch.setChecked(  # type: ignore[attr-defined]
                bool(preferences.get("auto_pipeline", True))
            )
        except Exception:  # noqa: BLE001
            pass

        # Status bar
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        self.status_label = QLabel("Iniciando AquaCNN…")
        self.status_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        bar.addWidget(self.status_label, 1)
        self.path_label = QLabel(str(self.dataset_path))
        self.path_label.setStyleSheet(f"color: {MUTED_2}; font-size: 10px;")
        bar.addPermanentWidget(self.path_label)
        self.setStatusBar(bar)

        self.toast_widget = Toast(central)
        self._toast = self.toast_widget

    def _shortcuts(self) -> None:
        for i in range(len(self.TITLES)):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.goto_page(idx))
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.choose_dataset_folder)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.rescan_dataset)
        QShortcut(QKeySequence("F5"), self, activated=self.explorer._run_pipeline)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._export_shortcut)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Esc"), self, activated=self.exit_fullscreen)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()

    def is_training(self) -> bool:
        thread = getattr(getattr(self, "training", None), "thread", None)
        return bool(thread and thread.isRunning())

    def restart_application(self) -> None:
        """Reinicia el proceso para aplicar la paleta completa."""
        if self.is_training():
            self.toast("Detén el entrenamiento antes de reiniciar la aplicación.", "warning")
            return
        self._save_cfg()
        from PySide6.QtCore import QProcess

        program = sys.executable
        if getattr(sys, "frozen", False):
            args: list[str] = []
        else:
            args = [str(Path(__file__).resolve().parent.parent / "app.py")]
        started = QProcess.startDetached(program, args, str(Path.cwd()))
        if started:
            QApplication.quit()
        else:
            self.toast("No se pudo reiniciar automáticamente. Cierra y abre AquaCNN.", "warning")

    def _export_shortcut(self) -> None:
        self.goto_page(4)
        if self.training_result:
            self.results.export_dataset()

    def _on_training_config(self, cfg: TrainingConfig) -> None:
        self.training_cfg = cfg
        self.explorer.cfg.net_input_size = int(cfg.input_size)
        self.architecture.refresh_from_training()

    # ------------------------------------------------------------------
    def goto_page(self, index: int) -> None:
        index = max(0, min(len(self.TITLES) - 1, int(index)))
        self.stack.setCurrentIndex(index)
        self.sidebar.set_current(index)
        title, subtitle = self.TITLES[index]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)
        self.settings.setValue("last_page", index)
        if index == 2:
            self.architecture.refresh_from_training()
        elif index == 6:
            self.settings_page.refresh()

    def open_explorer(self, path: str | None = None) -> None:
        self.goto_page(1)
        if path:
            self.explorer.set_sample(path)

    def toast(self, message: str, kind: str = "info") -> None:
        self.toast_widget.show_message(message, kind)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "toast_widget"):
            self.toast_widget._reposition()

    # ------------------------------------------------------------------
    def load_dataset(self, path: str | Path, initial: bool = False) -> None:
        path = Path(path)
        try:
            ensure_structure(path)
        except OSError as exc:
            self.toast(f"No se pudo crear la estructura del dataset: {exc}", "error")
            return
        self.dataset_path = path
        self.path_label.setText(str(path))
        self.settings.setValue("dataset_path", str(path))
        self._scan_token += 1
        token = self._scan_token
        self.set_status(f"Escaneando {path.name}…")
        worker = ScanWorker(path)
        self._scan_workers.append(worker)
        worker.signals.done.connect(lambda info, thumbs, t=token: self._on_scan(info, thumbs, t))
        worker.signals.error.connect(lambda msg: self.toast(f"Error al escanear: {msg}", "error"))
        worker.signals.done.connect(lambda *_: self._release_worker(self._scan_workers, worker))
        worker.signals.error.connect(lambda *_: self._release_worker(self._scan_workers, worker))
        self.thread_pool.start(worker)

    @staticmethod
    def _release_worker(bucket: list, worker) -> None:
        try:
            bucket.remove(worker)
        except ValueError:
            pass

    def rescan_dataset(self) -> None:
        self.load_dataset(self.dataset_path)

    def _on_scan(self, info: DatasetInfo, thumbs: dict, token: int) -> None:
        if token != self._scan_token:
            return
        self.dataset_info = info
        self.dataset_thumbs = thumbs
        if info.total > 0:
            self.splits = make_splits(info, self.training_cfg)
        else:
            self.splits = None
        if (
            not self._default_evaluated
            and self.loaded_model is not None
            and self.splits
            and self.splits.get("test")
        ):
            # Evalúa una sola vez el modelo incluido para que la página de
            # Resultados tenga contenido desde el primer arranque.
            self._default_evaluated = True
            QTimer.singleShot(0, lambda: self.evaluate_loaded_model(silent=True))
        self.dashboard.refresh(info, thumbs, self.splits)
        self.explorer.refresh(info, thumbs)
        self.training.refresh(info, self.splits)
        self.architecture.refresh_model()
        self.sidebar.set_dataset(info.total, info.counts)
        self.dataset_chip.setText(f"{info.total} imágenes")
        self.settings_page.path_label.setText(str(self.dataset_path))
        self.dataset_chip.setIcon(icons.icon("database", ACCENT if info.total else MUTED, 16))
        if info.total == 0:
            self.set_status("Dataset vacío · agrega imágenes para comenzar")
        else:
            self.set_status(
                f"{info.total} imágenes · {info.class_count}/3 clases · "
                f"balance {info.balance * 100:.0f}% · train {len(self.splits['train'])} / "
                f"val {len(self.splits['val'])} / test {len(self.splits['test'])}"
            )
        if info.truncated:
            self.toast(
                f"Se encontraron {info.found} imágenes; se usarán {info.total} "
                f"(tope de {info.limit}).",
                "warning",
            )
            self.set_status(
                f"Tope alcanzado: {info.total} de {info.found} imágenes · "
                f"ajusta las carpetas para procesar el resto"
            )

    # ------------------------------------------------------------------
    def choose_dataset_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta del dataset", str(self.dataset_path))
        if not folder:
            return
        self.load_dataset(folder)
        self.toast("Carpeta del dataset actualizada.", "success")

    def open_dataset_folder(self) -> None:
        try:
            os.startfile(str(self.dataset_path))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            self.toast("No se pudo abrir la carpeta.", "warning")

    def create_structure(self) -> None:
        ensure_structure(self.dataset_path)
        self.toast("Estructura creada: Dataset Imagenes/trucha, mojarra y pavon.", "success")
        self.rescan_dataset()

    def use_bundled_dataset(self) -> None:
        bundled = paths.resource_path("Dataset Imagenes")
        if not bundled.exists():
            self.toast("No se encontró un dataset de ejemplo incluido.", "warning")
            return
        target = self.dataset_path
        copied = 0
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if bundled.resolve() != target.resolve():
                ensure_structure(target)
                for cls in CLASSES:
                    src = bundled / cls
                    if not src.exists():
                        continue
                    for f in src.iterdir():
                        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            dest = target / cls / f.name
                            if not dest.exists():
                                try:
                                    shutil.copy2(f, dest)
                                    copied += 1
                                except OSError:
                                    continue
        finally:
            QApplication.restoreOverrideCursor()
        self.rescan_dataset()
        if copied:
            self.toast(f"Dataset de ejemplo copiado ({copied} imágenes).", "success")
        else:
            self.toast("El dataset de ejemplo ya está disponible.", "info")

    def add_images_dialog(self, class_name: str) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, f"Añadir imágenes · {CLASS_LABELS[class_name]}",
            str(self.dataset_path), "Imágenes (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not files:
            return
        copied = import_images(files, class_name, self.dataset_path, MAX_DATASET_IMAGES)
        if len(copied) < len(files):
            self.toast(
                f"{len(copied)} imágenes añadidas · tope de {MAX_DATASET_IMAGES} alcanzado.",
                "warning",
            )
        else:
            self.toast(f"{len(copied)} imágenes añadidas a {CLASS_LABELS[class_name]}.", "success")
        self.rescan_dataset()

    def import_dropped_images(self, paths_in: list[str]) -> None:
        if not paths_in:
            return
        groups: dict[str, list[str]] = {}
        other: list[str] = []
        for raw in paths_in:
            p = Path(raw)
            cls = next((c for c in CLASSES if c in [part.lower() for part in p.parts]), None)
            if cls:
                groups.setdefault(cls, []).append(raw)
            else:
                other.append(raw)
        if other:
            item, ok = QInputDialog.getItem(
                self, "Asignar clase",
                f"¿A qué clase pertenecen las {len(other)} imágenes seleccionadas?",
                [CLASS_LABELS[c] for c in CLASSES], 0, False,
            )
            if ok:
                cls = CLASSES[[CLASS_LABELS[c] for c in CLASSES].index(item)]
                groups.setdefault(cls, []).extend(other)
        total = 0
        for cls, files in groups.items():
            total += len(import_images(files, cls, self.dataset_path, MAX_DATASET_IMAGES))
        if total:
            if total < sum(len(v) for v in groups.values()):
                self.toast(
                    f"{total} imágenes importadas · tope de {MAX_DATASET_IMAGES} alcanzado.",
                    "warning",
                )
            else:
                self.toast(f"{total} imágenes importadas al dataset.", "success")
            self.rescan_dataset()
        else:
            self.toast("No se importó ninguna imagen.", "warning")

    def apply_split(self, train: float, val: float, test: float, seed: int) -> None:
        if train + val + test <= 0:
            self.toast("Las proporciones deben sumar más de 0 %.", "warning")
            return
        self.training_cfg.train_ratio = train
        self.training_cfg.val_ratio = val
        self.training_cfg.test_ratio = test
        self.training_cfg.seed = seed
        self._save_cfg()
        if self.dataset_info and self.dataset_info.total:
            self.splits = make_splits(self.dataset_info, self.training_cfg)
            self.dashboard.refresh(self.dataset_info, self.dataset_thumbs, self.splits)
            self.training.refresh(self.dataset_info, self.splits)
            self.set_status(
                f"División aplicada · train {len(self.splits['train'])} · "
                f"val {len(self.splits['val'])} · test {len(self.splits['test'])}"
            )
            self.toast("División del dataset aplicada.", "success")
        else:
            self.toast("Carga imágenes antes de dividir el dataset.", "warning")

    # ------------------------------------------------------------------
    def refresh_results(self, result: TrainingResult) -> None:
        self.results.refresh(result)

    def evaluate_loaded_model(self, silent: bool = False) -> None:
        if self.loaded_model is None or self.loaded_cfg is None:
            self.toast("Carga un modelo .npz primero.", "warning")
            return
        if not self.splits or not self.splits["test"]:
            if not silent:
                self.toast("No hay conjunto de prueba para evaluar.", "warning")
            return
        paths_in = [str(s.path) for s in self.splits["test"]]
        labels = [s.label for s in self.splits["test"]]
        self.set_status("Evaluando modelo cargado…")
        worker = PredictWorker(self.loaded_model, paths_in, self.loaded_cfg)
        self._predict_workers.append(worker)
        worker.signals.done.connect(
            lambda probs, pred: self._on_loaded_predictions(probs, pred, paths_in, labels, silent)
        )
        worker.signals.error.connect(lambda msg: self.toast(f"Error al evaluar: {msg}", "error"))
        worker.signals.done.connect(lambda *_: self._release_worker(self._predict_workers, worker))
        worker.signals.error.connect(lambda *_: self._release_worker(self._predict_workers, worker))
        self.thread_pool.start(worker)

    def _on_loaded_predictions(self, probs, pred, paths_in, labels, silent: bool = False) -> None:
        import numpy as np
        y_true = np.array(labels, dtype=np.int64)
        metrics = evaluate_predictions(y_true, pred, len(CLASSES))
        metrics["test_loss"] = 0.0
        extra = self.loaded_meta.get("extra", {})
        metrics["best_val_accuracy"] = float(extra.get("best_val_accuracy", extra.get("accuracy", 0.0)))
        history = extra.get("history", {})
        result = TrainingResult(
            model=self.loaded_model,
            config=self.loaded_cfg,
            history=history,
            metrics=metrics,
            y_true=y_true,
            y_pred=pred,
            probs=probs,
            test_paths=paths_in,
            test_labels=labels,
            best_epoch=int(extra.get("best_epoch", 0)),
            duration=float(extra.get("duration", 0.0)),
            train_count=len(self.splits["train"]) if self.splits else 0,
            val_count=len(self.splits["val"]) if self.splits else 0,
            test_count=len(labels),
        )
        self.training_result = result
        self.refresh_results(result)
        if silent:
            self.set_status(
                f"Modelo incluido · exactitud {metrics['accuracy'] * 100:.2f}% en el conjunto de prueba"
            )
        else:
            self.goto_page(4)
            self.toast(f"Modelo evaluado · exactitud {metrics['accuracy'] * 100:.2f}%", "success")
            self.set_status(f"Modelo cargado evaluado · accuracy {metrics['accuracy'] * 100:.2f}%")

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_cfg()
        self.settings.setValue("last_page", self.stack.currentIndex())
        thread = getattr(self.training, "thread", None)
        if thread is not None and thread.isRunning():
            thread.request_stop()
            if not thread.wait(6000):
                thread.terminate()
                thread.wait(1000)
        export_thread = getattr(self.results, "export_thread", None)
        if export_thread is not None and export_thread.isRunning():
            export_thread.request_stop()
            export_thread.wait(2500)
        super().closeEvent(event)
