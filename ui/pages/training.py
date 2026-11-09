"""Página de configuración y monitoreo del entrenamiento en vivo."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.config import CLASS_LABELS, CLASSES, EQUALIZATIONS, TrainingConfig
from core.dataset import DatasetInfo
from core.model_io import load_model, save_model
from ui import icons
from ui.theme import (
    ACCENT,
    ACCENT_2,
    BG_ALT,
    DANGER_TEXT,
    MUTED,
    ON_ACCENT,
    PURPLE,
    SUCCESS,
    TEXT,
    WARNING,
)
from ui.glossary import GlossaryLabel
from ui.widgets.cards import Badge, Card, Divider, SectionHeader, ToggleSwitch
from ui.widgets.charts import TrainingChart
from ui.workers import TrainingThread


class MetricChip(Card):
    def __init__(self, label: str, value: str = "—", color: str = ACCENT, parent: QWidget | None = None):
        super().__init__(parent, padding=10, spacing=1)
        self.setMinimumWidth(92)
        self.value = QLabel(value)
        self.value.setStyleSheet(f"font-size: 17px; font-weight: 700; color: {color};")
        self.label = GlossaryLabel(label)
        self.label.setProperty("subtle", True)
        self.label.setStyleSheet("font-size: 9px;")
        self.body().addWidget(self.value)
        self.body().addWidget(self.label)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class TrainingPage(QWidget):
    configChanged = Signal(object)

    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent)
        self.main = main
        self.thread: TrainingThread | None = None
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}
        self._info: DatasetInfo | None = None
        self._splits = None
        self._active_epochs = 30

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ---------------- Configuración ----------------
        cfg_scroll = QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setMinimumWidth(330)
        cfg_scroll.setMaximumWidth(380)
        cfg_host = QWidget()
        cfg_host.setObjectName("Page")
        cl = QVBoxLayout(cfg_host)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)
        cfg_scroll.setWidget(cfg_host)
        # (el panel de acciones se añade como pie fijo más abajo)

        data_card = Card(padding=14, spacing=8)
        data_card.body().addWidget(SectionHeader("Conjunto de datos", "División estratificada"))
        self.split_chips = QHBoxLayout()
        self.chip_train = Badge("Train 0", ACCENT_2)
        self.chip_val = Badge("Val 0", SUCCESS)
        self.chip_test = Badge("Test 0", WARNING)
        for c in (self.chip_train, self.chip_val, self.chip_test):
            self.split_chips.addWidget(c)
        self.chip_train.setToolTip("Imágenes usadas para ajustar los pesos de la red.")
        self.chip_val.setToolTip("Imágenes reservadas para vigilar el sobreajuste en cada época.")
        self.chip_test.setToolTip("Imágenes que la red nunca ve al entrenar; miden la generalización final.")
        self.split_chips.addStretch(1)
        data_card.body().addLayout(self.split_chips)
        self.data_info = QLabel("Sin dataset cargado")
        self.data_info.setProperty("subtle", True)
        self.data_info.setWordWrap(True)
        self.data_info.setToolTip("Resumen del dataset escaneado y de la división estratificada por clase.")
        data_card.body().addWidget(self.data_info)
        cl.addWidget(data_card)

        model_card = Card(padding=14, spacing=8)
        model_card.body().addWidget(SectionHeader("Clasificador Neuronal", "Red Neuronal Convolucional (CNN)"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_cnn = Badge("Red Neuronal Convolucional (CNN)", ACCENT)
        self.btn_cnn.setToolTip("Modelo fijo del laboratorio: CNN con convoluciones, pooling y capas ocultas.")
        row.addWidget(self.btn_cnn)
        row.addStretch(1)
        model_card.body().addLayout(row)
        self.arch_info = QLabel(
            "Conv 3×3 → ReLU → MaxPool → Conv 3×3 → ReLU → MaxPool → Flatten → Capa oculta → Softmax"
        )
        self.arch_info.setProperty("subtle", True)
        self.arch_info.setWordWrap(True)
        self.arch_info.setToolTip(
            "Flujo de la red: dos bloques convolucionales, aplanado y una capa oculta con ReLU antes de las 3 clases."
        )
        model_card.body().addWidget(self.arch_info)
        self.input_combo = QComboBox()
        for size in (32, 40, 48, 64):
            self.input_combo.addItem(f"{size} × {size}", size)
        self.input_combo.setCurrentIndex(2)
        self.input_combo.setToolTip(
            "Resolución a la que se redimensiona cada imagen antes de la red. "
            "Más grande conserva detalle, pero multiplica el coste de cálculo."
        )
        model_card.body().addLayout(self._combo("Tamaño de entrada", self.input_combo))
        hrow = QHBoxLayout()
        hrow.setSpacing(8)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(8, 512)
        self.hidden_spin.setValue(64)
        self.hidden_spin.setSingleStep(8)
        self.f1_spin = QSpinBox()
        self.f1_spin.setRange(2, 32)
        self.f1_spin.setValue(6)
        self.f2_spin = QSpinBox()
        self.f2_spin.setRange(2, 64)
        self.f2_spin.setValue(12)
        self.hidden_spin.setToolTip(
            "Neuronas de la capa oculta: cuántos patrones puede combinar la red antes de decidir la clase."
        )
        self.f1_spin.setToolTip(
            "Filtros de la primera convolución: cada filtro aprende un patrón local distinto (bordes, texturas)."
        )
        self.f2_spin.setToolTip(
            "Filtros de la segunda convolución: combinan los patrones simples de la primera capa en estructuras más complejas."
        )
        hrow.addLayout(self._combo("Neuronas ocultas", self.hidden_spin))
        hrow.addLayout(self._combo("Filtros 1", self.f1_spin))
        hrow.addLayout(self._combo("Filtros 2", self.f2_spin))
        model_card.body().addLayout(hrow)
        cl.addWidget(model_card)

        train_card = Card(padding=14, spacing=8)
        train_card.body().addWidget(SectionHeader("Hiperparámetros", "Backpropagation + Momentum"))
        erow = QHBoxLayout()
        erow.setSpacing(8)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 300)
        self.epochs_spin.setValue(30)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(2, 128)
        self.batch_spin.setValue(16)
        self.batch_spin.setSingleStep(2)
        self.epochs_spin.setToolTip(
            "Época = una pasada completa por todas las imágenes de entrenamiento. Más épocas pueden mejorar "
            "hasta que aparece sobreajuste."
        )
        self.batch_spin.setToolTip(
            "Batch = imágenes que se procesan juntas antes de actualizar los pesos. Batches pequeños "
            "hacen el aprendizaje más ruidoso pero suelen generalizar mejor."
        )
        erow.addLayout(self._combo("Épocas", self.epochs_spin))
        erow.addLayout(self._combo("Batch", self.batch_spin))
        train_card.body().addLayout(erow)
        lrow = QHBoxLayout()
        lrow.setSpacing(8)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.5)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setDecimals(2)
        self.momentum_spin.setRange(0.0, 0.99)
        self.momentum_spin.setSingleStep(0.05)
        self.momentum_spin.setValue(0.90)
        self.lr_spin.setToolTip(
            "Learning rate: tamaño del paso al corregir los pesos. Muy alto diverge; muy bajo aprende muy lento."
        )
        self.momentum_spin.setToolTip(
            "Momentum: acumula la dirección de los gradientes anteriores para avanzar más rápido y con menos oscilación."
        )
        lrow.addLayout(self._combo("Learning rate", self.lr_spin))
        lrow.addLayout(self._combo("Momentum", self.momentum_spin))
        train_card.body().addLayout(lrow)
        wrow = QHBoxLayout()
        wrow.setSpacing(8)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setDecimals(5)
        self.wd_spin.setRange(0.0, 0.1)
        self.wd_spin.setSingleStep(0.0001)
        self.wd_spin.setValue(0.0001)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.wd_spin.setToolTip(
            "Weight decay (penalización L2): mantiene los pesos pequeños para reducir el sobreajuste."
        )
        self.seed_spin.setToolTip(
            "Semilla aleatoria: fija la inicialización y el orden de las imágenes para que el experimento sea reproducible."
        )
        wrow.addLayout(self._combo("Weight decay", self.wd_spin))
        wrow.addLayout(self._combo("Semilla", self.seed_spin))
        train_card.body().addLayout(wrow)
        self.aug_toggle = self._toggle_row("Data augmentation", True)
        self.weights_toggle = self._toggle_row("Pesos por clase", True)
        self.roi_toggle = self._toggle_row("Recorte ROI al entrenar", True)
        self.aug_toggle.setToolTip(
            "Data augmentation: genera volteos, cambios de brillo y recortes para que la red generalice mejor."
        )
        self.weights_toggle.setToolTip(
            "Pesos por clase: compensa el desbalance dando más importancia a las clases con menos imágenes."
        )
        self.roi_toggle.setToolTip(
            "Recorte ROI: centra un recorte cuadrado en el pez antes de redimensionar, igual que en el pipeline."
        )
        train_card.body().addWidget(self.aug_toggle)
        train_card.body().addWidget(self.weights_toggle)
        train_card.body().addWidget(self.roi_toggle)
        self.equal_combo = QComboBox()
        for key, label in EQUALIZATIONS.items():
            self.equal_combo.addItem(label, key)
        self.equal_combo.setCurrentIndex(2)
        self.equal_combo.setToolTip(
            "Mejora de contraste aplicada a cada imagen: ninguna, ecualización global o CLAHE adaptativa por bloques."
        )
        train_card.body().addLayout(self._combo("Mejora de contraste", self.equal_combo))
        cl.addWidget(train_card)

        actions = Card(padding=14, spacing=8)
        self.preset_btn = QPushButton("Aplicar configuración recomendada")
        self.preset_btn.setProperty("variant", "ghost")
        self.preset_btn.setIcon(icons.icon("check", MUTED, 15))
        self.preset_btn.setToolTip(
            "Ajusta 48×48, 6/12 filtros, 64 neuronas ocultas, 80 épocas, lr 0.01, "
            "weight decay 1e-4, CLAHE, aumento y pesos por clase: la combinación que mejor "
            "equilibró las tres clases en este laboratorio."
        )
        self.preset_btn.clicked.connect(self._apply_recommended)
        actions.body().addWidget(self.preset_btn)

        self.start_btn = QPushButton("Iniciar entrenamiento")
        self.start_btn.setProperty("variant", "primary")
        self.start_btn.setIcon(icons.icon("play", ON_ACCENT, 17))
        self.stop_btn = QPushButton("Detener")
        self.stop_btn.setProperty("variant", "danger")
        self.stop_btn.setIcon(icons.icon("stop", DANGER_TEXT, 15))
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("Guardar modelo")
        self.save_btn.setProperty("variant", "ghost")
        self.save_btn.setIcon(icons.icon("save", MUTED, 15))
        self.save_btn.setEnabled(False)
        self.load_btn = QPushButton("Cargar modelo")
        self.load_btn.setProperty("variant", "ghost")
        self.load_btn.setIcon(icons.icon("folder", MUTED, 15))
        self.start_btn.setToolTip("Comienza el entrenamiento con la configuración actual (Backpropagation + Momentum).")
        self.stop_btn.setToolTip("Detiene el entrenamiento de forma segura conservando el último estado.")
        self.save_btn.setToolTip("Guarda pesos, configuración e historial en un archivo .npz.")
        self.load_btn.setToolTip("Carga un modelo .npz guardado previamente para evaluarlo o seguir entrenándolo.")
        actions.body().addWidget(self.start_btn)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.stop_btn)
        row2.addWidget(self.save_btn)
        actions.body().addLayout(row2)
        actions.body().addWidget(self.load_btn)
        self.status = QLabel("Configura el modelo y pulsa iniciar.")
        self.status.setProperty("subtle", True)
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(30)
        actions.body().addWidget(self.status)
        cl.addStretch(1)

        left_wrap = QWidget()
        lw = QVBoxLayout(left_wrap)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setSpacing(10)
        lw.addWidget(cfg_scroll, 1)
        lw.addWidget(actions, 0)
        splitter.insertWidget(0, left_wrap)

        # ---------------- Monitoreo ----------------
        monitor = QWidget()
        ml = QVBoxLayout(monitor)
        ml.setContentsMargins(4, 0, 0, 0)
        ml.setSpacing(10)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chip_epoch = MetricChip("Época", "0 / 0", ACCENT)
        self.chip_loss = MetricChip("Loss", "—", ACCENT_2)
        self.chip_vloss = MetricChip("Val loss", "—", WARNING)
        self.chip_acc = MetricChip("Accuracy", "—", SUCCESS)
        self.chip_vacc = MetricChip("Val accuracy", "—", ACCENT)
        self.chip_eta = MetricChip("ETA", "—", PURPLE)
        for c in (self.chip_epoch, self.chip_loss, self.chip_vloss, self.chip_acc, self.chip_vacc, self.chip_eta):
            chips.addWidget(c, 1)
        self.chip_epoch.setToolTip("Época actual y total configurado.")
        self.chip_loss.setToolTip("Pérdida de entrenamiento (entropía cruzada); debe bajar y estabilizarse.")
        self.chip_vloss.setToolTip("Pérdida de validación; si sube mientras la de entrenamiento baja, hay sobreajuste.")
        self.chip_acc.setToolTip("Exactitud sobre las imágenes de entrenamiento.")
        self.chip_vacc.setToolTip("Exactitud sobre validación: la señal más útil para elegir la mejor época.")
        self.chip_eta.setToolTip("Tiempo estimado restante para terminar el entrenamiento.")
        ml.addLayout(chips)

        chart_card = Card(padding=12, spacing=6)
        self.chart = TrainingChart()
        self.chart.setMinimumHeight(360)
        self.chart.setToolTip("Evolución de pérdida y exactitud por época (entrenamiento vs validación).")
        chart_card.body().addWidget(self.chart)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Listo para entrenar")
        self.progress.setToolTip("Progreso del entrenamiento; se detiene al terminar todas las épocas o al pulsar Detener.")
        chart_card.body().addWidget(self.progress)
        ml.addWidget(chart_card, 1)

        log_card = Card(padding=12, spacing=6)
        log_head = SectionHeader("Registro de entrenamiento", "Backpropagation · Momentum · Softmax")
        self.clear_log_btn = QPushButton("Limpiar")
        self.clear_log_btn.setProperty("variant", "ghost")
        self.clear_log_btn.setStyleSheet("padding: 4px 10px; font-size: 11px;")
        self.clear_log_btn.setToolTip("Vacía el registro de texto sin afectar al modelo entrenado.")
        log_head.add_right(self.clear_log_btn)
        log_card.body().addWidget(log_head)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(170)
        self.log.setToolTip("Registro detallado: época, pérdidas, exactitud, mejor modelo y avisos del entrenamiento.")
        self.log.setStyleSheet(
            "font-family: 'Cascadia Code','Consolas',monospace; font-size: 10.5px;"
            f"background: {BG_ALT}; border-radius: 9px;"
        )
        log_card.body().addWidget(self.log)
        ml.addWidget(log_card)
        splitter.addWidget(monitor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 1000])

        # Conexiones
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.save_btn.clicked.connect(self.save_trained_model)
        self.load_btn.clicked.connect(self.load_trained_model)
        self.clear_log_btn.clicked.connect(self.log.clear)
        self._set_model("cnn")
        self.input_combo.currentIndexChanged.connect(self._emit_config)
        for spin in (self.hidden_spin, self.f1_spin, self.f2_spin):
            spin.valueChanged.connect(self._emit_config)
        for spin in (self.epochs_spin, self.batch_spin, self.seed_spin):
            spin.valueChanged.connect(self._emit_config)
        for dspin in (self.lr_spin, self.momentum_spin, self.wd_spin):
            dspin.valueChanged.connect(self._emit_config)
        self.equal_combo.currentIndexChanged.connect(self._emit_config)
        for toggle in (self.aug_toggle, self.weights_toggle, self.roi_toggle):
            toggle._switch.toggled.connect(self._emit_config)  # type: ignore[attr-defined]
        self._apply_config(self.main.training_cfg)

    def _apply_config(self, cfg: TrainingConfig) -> None:
        """Refleja la configuración guardada sin disparar señales en cascada."""
        widgets = (
            self.input_combo, self.hidden_spin, self.f1_spin, self.f2_spin,
            self.epochs_spin, self.batch_spin, self.seed_spin,
            self.lr_spin, self.momentum_spin, self.wd_spin, self.equal_combo,
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            idx = self.input_combo.findData(int(cfg.input_size))
            if idx >= 0:
                self.input_combo.setCurrentIndex(idx)
            self.hidden_spin.setValue(int(cfg.hidden_units))
            self.f1_spin.setValue(int(cfg.conv_filters))
            self.f2_spin.setValue(int(cfg.conv_filters2))
            self.epochs_spin.setValue(int(cfg.epochs))
            self.batch_spin.setValue(int(cfg.batch_size))
            self.seed_spin.setValue(int(cfg.seed))
            self.lr_spin.setValue(float(cfg.learning_rate))
            self.momentum_spin.setValue(float(cfg.momentum))
            self.wd_spin.setValue(float(cfg.weight_decay))
            idx = self.equal_combo.findData(cfg.equalization)
            if idx >= 0:
                self.equal_combo.setCurrentIndex(idx)
            self.aug_toggle._switch.setChecked(bool(cfg.augmentation))  # type: ignore[attr-defined]
            self.weights_toggle._switch.setChecked(bool(cfg.class_weights))  # type: ignore[attr-defined]
            self.roi_toggle._switch.setChecked(bool(cfg.roi_crop))  # type: ignore[attr-defined]
        finally:
            for w in widgets:
                w.blockSignals(False)

    # ------------------------------------------------------------------
    def _emit_config(self, *_args) -> None:
        self.configChanged.emit(self._config())

    def _combo(self, text: str, widget: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(3)
        lbl = GlossaryLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 10px;")
        if widget.toolTip():
            lbl.setToolTip(widget.toolTip())
        box.addWidget(lbl)
        box.addWidget(widget)
        return box

    def _toggle_row(self, text: str, checked: bool) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = GlossaryLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 11px;")
        row.addWidget(lbl, 1)
        sw = ToggleSwitch(checked)
        row.addWidget(sw)
        w._switch = sw  # type: ignore[attr-defined]
        return w

    def _set_model(self, kind: str = "cnn") -> None:
        self.f1_spin.setEnabled(True)
        self.f2_spin.setEnabled(True)
        self.arch_info.setText(
            "Conv 3×3 → ReLU → MaxPool → Conv 3×3 → ReLU → MaxPool → Flatten → Capa oculta → Softmax"
        )
        self.configChanged.emit(self._config())

    def _config(self) -> TrainingConfig:
        return TrainingConfig(
            model_type="cnn",
            input_size=int(self.input_combo.currentData()),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=self.lr_spin.value(),
            momentum=self.momentum_spin.value(),
            weight_decay=self.wd_spin.value(),
            hidden_units=self.hidden_spin.value(),
            conv_filters=self.f1_spin.value(),
            conv_filters2=self.f2_spin.value(),
            augmentation=self.aug_toggle._switch.isChecked(),  # type: ignore[attr-defined]
            class_weights=self.weights_toggle._switch.isChecked(),  # type: ignore[attr-defined]
            roi_crop=self.roi_toggle._switch.isChecked(),  # type: ignore[attr-defined]
            seed=self.seed_spin.value(),
            equalization=self.equal_combo.currentData(),
            clahe_clip=self.main.pipeline_cfg.clahe_clip,
            train_ratio=self.main.training_cfg.train_ratio,
            val_ratio=self.main.training_cfg.val_ratio,
            test_ratio=self.main.training_cfg.test_ratio,
        )

    def _apply_recommended(self) -> None:
        """Carga el preset que mejor equilibró trucha, mojarra y pavón."""
        cfg = self._config()
        cfg.input_size = 48
        cfg.hidden_units = 64
        cfg.conv_filters = 6
        cfg.conv_filters2 = 12
        cfg.epochs = 80
        cfg.batch_size = 16
        cfg.learning_rate = 0.01
        cfg.momentum = 0.9
        cfg.weight_decay = 1e-4
        cfg.seed = 42
        cfg.augmentation = True
        cfg.class_weights = True
        cfg.roi_crop = True
        cfg.equalization = "clahe"
        cfg.dropout = 0.25
        cfg.label_smoothing = 0.05
        self._apply_config(cfg)
        self._emit_config()
        self.status.setText(
            "Configuración recomendada aplicada · 48×48 · 6/12 filtros · 64 ocultas · "
            "80 épocas · lr 0.01 · CLAHE · ROI · aumento · pesos por clase."
        )
        self._log(
            "Preset recomendado aplicado. Entrena y compara val accuracy antes de guardar; "
            "si baja, vuelve a 30-60 épocas."
        )

    def set_loaded_model(self, meta: dict, name: str) -> None:
        """Refleja en la interfaz el modelo preentrenado cargado al arrancar."""
        extra = meta.get("extra", {}) or {}
        acc = extra.get("accuracy")
        text = f"Modelo incluido cargado: {name}"
        if acc is not None:
            text += f" · exactitud de entrenamiento {float(acc) * 100:.2f}%"
        self.status.setText(text)

    def refresh(self, info: DatasetInfo, splits) -> None:
        self._info = info
        self._splits = splits
        if splits:
            self.chip_train.setText(f"Train {len(splits['train'])}")
            self.chip_val.setText(f"Val {len(splits['val'])}")
            self.chip_test.setText(f"Test {len(splits['test'])}")
            self.data_info.setText(
                f"{info.total} imágenes · {info.class_count} clases · semilla {self.main.training_cfg.seed}"
            )
        else:
            self.chip_train.setText("Train 0")
            self.chip_val.setText("Val 0")
            self.chip_test.setText("Test 0")
            self.data_info.setText("Carga el dataset y aplica la división en Inicio.")

    def _log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {msg}")
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_training(self) -> None:
        if self.thread and self.thread.isRunning():
            return
        if not self._info or self._info.is_empty:
            self.main.toast("No hay imágenes en el dataset. Carga el dataset primero.", "warning")
            return
        if not self._splits or not self._splits["train"]:
            self.main.toast("Aplica la división del dataset en la página de Inicio.", "warning")
            return
        cfg = self._config()
        self.main.training_cfg = cfg
        self.main._save_cfg()
        self._active_epochs = cfg.epochs
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}
        self.chart._reset()
        self.progress.setValue(0)
        self.progress.setFormat("Preparando datos…")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.status.setText("Entrenando en segundo plano…")
        self._log(
            f"Inicio · modelo={cfg.model_type.upper()} · input={cfg.input_size}px · "
            f"epochs={cfg.epochs} · batch={cfg.batch_size} · lr={cfg.learning_rate} · momentum={cfg.momentum}"
        )
        self.thread = TrainingThread(self._splits["train"], self._splits["val"], self._splits["test"], cfg, self)
        self.thread.log.connect(self._log)
        self.thread.epoch.connect(self.on_epoch)
        self.thread.batch.connect(self.on_batch)
        self.thread.done.connect(self.on_done)
        self.thread.failed.connect(self.on_failed)
        self.thread.start()

    def stop_training(self) -> None:
        if self.thread and self.thread.isRunning():
            self.thread.request_stop()
            self._log("Solicitando detención…")
            self.stop_btn.setEnabled(False)

    def on_batch(self, epoch: int, batch: int, total: int, loss: float) -> None:
        total_epochs = self._active_epochs
        frac = ((epoch - 1) + batch / max(1, total)) / max(1, total_epochs)
        self.progress.setValue(int(frac * 100))
        self.progress.setFormat(f"Época {epoch}/{total_epochs} · batch {batch}/{total} · loss {loss:.4f}")

    def on_epoch(self, info: dict) -> None:
        for key in ("loss", "accuracy", "val_loss", "val_accuracy", "lr"):
            self.history[key].append(float(info.get(key, 0.0)))
        epoch = int(info.get("epoch", len(self.history["loss"])))
        epochs = int(info.get("epochs", self._active_epochs or epoch or 1))
        loss = float(info.get("loss", 0.0))
        val_loss = float(info.get("val_loss", 0.0))
        acc = float(info.get("accuracy", 0.0))
        val_acc = float(info.get("val_accuracy", 0.0))
        self.chart.update_history(self.history, int(info.get("best_epoch", 0)))
        self.chip_epoch.set_value(f"{epoch} / {epochs}")
        self.chip_loss.set_value(f"{loss:.4f}")
        self.chip_vloss.set_value(f"{val_loss:.4f}")
        self.chip_acc.set_value(f"{acc * 100:.1f}%")
        self.chip_vacc.set_value(f"{val_acc * 100:.1f}%")
        eta = float(info.get("eta", 0.0))
        self.chip_eta.set_value(f"{int(eta)}s" if eta < 120 else f"{eta / 60:.1f}m")
        self.progress.setValue(int(epoch / max(1, epochs) * 100))
        self.progress.setFormat(
            f"Época {epoch}/{epochs} · acc {acc * 100:.1f}% · val_acc {val_acc * 100:.1f}%"
        )

    def on_done(self, result) -> None:
        self.main.training_result = result
        self.main.refresh_results(result)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.status.setText(
            f"Entrenamiento finalizado · exactitud de prueba {result.metrics['accuracy'] * 100:.2f}% · "
            f"mejor época {result.best_epoch} · {result.duration:.1f}s"
        )
        self.progress.setValue(100)
        self.progress.setFormat(f"Completado · test accuracy {result.metrics['accuracy'] * 100:.2f}%")
        self._log(
            f"Listo · test accuracy={result.metrics['accuracy'] * 100:.2f}% · "
            f"F1 macro={result.metrics['macro_f1'] * 100:.2f}% · duración={result.duration:.1f}s"
        )
        self.main.toast("Entrenamiento completado. Abriendo resultados…", "success")
        QTimer.singleShot(900, lambda: self.main.goto_page(4))

    def on_failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Error durante el entrenamiento.")
        self.progress.setFormat("Error")
        self._log("ERROR: " + message)
        self.main.toast("Error en el entrenamiento: " + message.splitlines()[0], "error")

    def save_trained_model(self) -> None:
        result = self.main.training_result
        if not result:
            self.main.toast("Todavía no hay un modelo entrenado.", "warning")
            return
        default = self.main.models_dir / f"AquaCNN_{result.config.model_type}_{datetime.now():%Y%m%d_%H%M}.npz"
        path, _ = QFileDialog.getSaveFileName(self, "Guardar modelo", str(default), "Modelo NumPy (*.npz)")
        if not path:
            return
        extra = {
            "accuracy": float(result.metrics.get("accuracy", 0.0)),
            "macro_f1": float(result.metrics.get("macro_f1", 0.0)),
            "best_epoch": int(result.best_epoch),
            "duration": float(result.duration),
            "train_count": int(result.train_count),
            "val_count": int(result.val_count),
            "test_count": int(result.test_count),
            "history": {k: [float(v) for v in vals] for k, vals in result.history.items()},
        }
        try:
            saved = save_model(path, result.model, result.config, extra)
            self._log(f"Modelo guardado en {saved}")
            self.main.toast("Modelo guardado correctamente.", "success")
        except Exception as exc:  # noqa: BLE001
            self.main.toast(f"No se pudo guardar el modelo: {exc}", "error")

    def load_trained_model(self) -> None:
        start = self.main.models_dir
        path, _ = QFileDialog.getOpenFileName(self, "Cargar modelo", str(start), "Modelo NumPy (*.npz)")
        if not path:
            return
        try:
            model, meta, cfg = load_model(path)
            self.main.loaded_model = model
            self.main.loaded_meta = meta
            self.main.loaded_cfg = cfg
            self.main.loaded_path = Path(path)
            self.main._default_evaluated = True
            self._log(f"Modelo cargado: {Path(path).name} · {meta.get('model_type', '?')}")
            self.main.evaluate_loaded_model()
        except Exception as exc:  # noqa: BLE001
            self.main.toast(f"No se pudo cargar el modelo: {exc}", "error")
