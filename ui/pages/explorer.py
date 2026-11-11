"""Explorador visual multi-etapa con controles en caliente."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.config import (
    CLASSES,
    CLASS_COLORS,
    CLASS_LABELS,
    EQUALIZATIONS,
    KERNELS,
    POOLINGS,
    PipelineConfig,
)
from core.dataset import DatasetInfo
from ui import icons
from ui.theme import (
    ACCENT,
    BORDER,
    DANGER,
    MUTED,
    MUTED_2,
    ON_ACCENT,
    PURPLE,
    SUCCESS,
    SURFACE_2,
    TEXT,
    rgba,
)
from ui.widgets.cards import Badge, Card, Divider, ElidedLabel, SectionHeader, ToggleSwitch
from ui.widgets.charts import HistogramCanvas
from ui.widgets.flow_layout import ResponsiveGrid
from ui.widgets.image_view import ImageView, open_image_dialog
from ui.widgets.kernel_editor import KernelEditor
from ui.workers import PipelineWorker


class StageCard(QFrame):
    toggled = Signal(bool)

    def __init__(
        self,
        number: int,
        title: str,
        subtitle: str,
        stage_key: str = "",
        can_toggle: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("StageCard")
        self.stage_key = stage_key
        self.can_toggle = can_toggle
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._title = title
        self._subtitle = subtitle
        self._image: np.ndarray | None = None
        self._canvas: QWidget | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 11, 12, 11)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        num = QLabel(str(number))
        num.setFixedSize(22, 22)
        num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num.setStyleSheet(
            f"background: {rgba(ACCENT, 0.10)}; color: {ACCENT}; border: 1px solid {rgba(ACCENT, 0.35)};"
            "border-radius: 7px; font-size: 10px; font-weight: 700;"
        )
        head.addWidget(num)
        col = QVBoxLayout()
        col.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet("font-size: 12px; font-weight: 700;")
        col.addWidget(t)
        s = QLabel(subtitle)
        s.setProperty("subtle", True)
        s.setStyleSheet("font-size: 9px;")
        col.addWidget(s)
        head.addLayout(col, 1)
        self.off_badge = QLabel("DESACTIVADA")
        self.off_badge.setStyleSheet(
            f"background: rgba(251,113,133,0.12); color: {DANGER}; border: 1px solid rgba(251,113,133,0.35);"
            "border-radius: 7px; padding: 1px 6px; font-size: 8px; font-weight: 700;"
        )
        self.off_badge.setVisible(False)
        head.addWidget(self.off_badge)

        self.switch = ToggleSwitch(checked=True)
        self.switch.setFixedSize(38, 22)
        self.switch.setEnabled(can_toggle)
        if not can_toggle:
            self.switch.setToolTip("Etapa base obligatoria del pipeline.")
        else:
            self.switch.setToolTip(f"Activar / desactivar etapa: {title}")
        self.switch.toggled.connect(self._on_switch_toggled)
        head.addWidget(self.switch)

        self.expand_btn = QPushButton()
        self.expand_btn.setProperty("variant", "icon")
        self.expand_btn.setIcon(icons.icon("maximize", MUTED_2, 14))
        self.expand_btn.setFixedSize(26, 26)
        self.expand_btn.setToolTip("Ampliar")
        head.addWidget(self.expand_btn)
        lay.addLayout(head)
        self.view = ImageView("Procesando…", "image", min_height=176, rounded=9)
        self.view.setFixedHeight(178)
        lay.addWidget(self.view)
        self.footer = QLabel("—")
        self.footer.setProperty("mono", True)
        self.footer.setWordWrap(True)
        lay.addWidget(self.footer)
        self.expand_btn.clicked.connect(self._expand)
        self.view.doubleClicked.connect(self._expand)

    def _on_switch_toggled(self, checked: bool) -> None:
        self.set_enabled(checked)
        self.toggled.emit(checked)

    def set_enabled(self, enabled: bool) -> None:
        self.off_badge.setVisible(not enabled)
        if self.switch.isChecked() != enabled:
            self.switch.blockSignals(True)
            self.switch.setChecked(enabled)
            self.switch.blockSignals(False)
        if enabled:
            self.setStyleSheet("")
            self.footer.setStyleSheet("")
        else:
            self.setStyleSheet(
                "QFrame#StageCard { border: 1px dashed rgba(251,113,133,0.55);"
                "background: rgba(251,113,133,0.04); border-radius: 14px; }"
            )
            self.footer.setStyleSheet(f"color: {DANGER};")
        self.setToolTip("" if enabled else f"Etapa '{self._title}' desactivada: la imagen pasa sin cambios a la siguiente etapa.")

    def set_image(self, img: np.ndarray | None, footer: str = "") -> None:
        self._image = img
        self.view.set_image(img)
        self.footer.setText(footer)

    def set_canvas(self, canvas: QWidget, footer: str = "") -> None:
        self._canvas = canvas
        self._image = None
        layout = self.layout()
        layout.replaceWidget(self.view, canvas)  # type: ignore[arg-type]
        self.view.deleteLater()
        canvas.setFixedHeight(178)
        self.footer.setText(footer)
        self.expand_btn.setEnabled(False)

    def _expand(self) -> None:
        if self._image is not None:
            open_image_dialog(self, self._image, self._title, self._subtitle)


class ControlsPanel(QWidget):
    changed = Signal()
    recalc = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Los interruptores de cada etapa viven en su propia tarjeta del pipeline.
        # Aquí solo se conserva el mapa ``key -> ToggleSwitch`` para config(),
        # set_stage_enabled() y la sincronización con la tarjeta.
        self.stage_switches: dict[str, ToggleSwitch] = {}

        # Color
        self.color_group = self._group("Espacio de color", "layers")
        self.color_buttons: dict[str, QPushButton] = {}
        self.color_btn_group = QButtonGroup(self)
        self.color_btn_group.setExclusive(True)
        row = QHBoxLayout()
        row.setSpacing(5)
        for key, label in (("rgb", "RGB"), ("gray", "Gris"), ("hsv", "HSV")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("variant", "chip")
            b.clicked.connect(lambda _=False, k=key: self._set_color(k))
            self.color_btn_group.addButton(b)
            self.color_buttons[key] = b
            row.addWidget(b)
        self.color_buttons["rgb"].setChecked(True)
        self.color_group.body().addLayout(row)
        self.show_channels = self._toggle_row("Mostrar canales R/G/B + HSV", True)
        self.color_group.body().addWidget(self.show_channels)
        self.conversion_toggle = self._toggle_row("Convertir a gris / HSV", True)
        self.color_group.body().addWidget(self.conversion_toggle)

        # Geometría
        self.geo_group = self._group("Geometría y normalización", "maximize")
        self.roi_toggle = self._toggle_row("Segmentación Otsu (contorno mayor)", True)
        self.geo_group.body().addWidget(self.roi_toggle)
        self.margin_slider, self.margin_label = self._slider_row("Margen de la caja", 0, 40, 15)
        self.geo_group.body().addLayout(self._labeled_row("Margen de la caja", self.margin_label, self.margin_slider))
        self.resize_toggle = self._toggle_row("Redimensionar", True)
        self.geo_group.body().addWidget(self.resize_toggle)
        self.resize_combo = QComboBox()
        for size in (64, 96, 128, 192, 256):
            self.resize_combo.addItem(f"{size} × {size}", size)
        self.resize_combo.setCurrentIndex(2)
        self.geo_group.body().addLayout(self._combo_row("Redimensionar a", self.resize_combo))
        self.normalize_toggle = self._toggle_row("Normalizar a [0, 1]", True)
        self.geo_group.body().addWidget(self.normalize_toggle)
        self.roi_toggle.setToolTip(
            "Umbral de Otsu → contorno más grande → caja envolvente: localiza el pez y elimina el fondo."
        )
        self.margin_slider.setToolTip(
            "Espacio extra alrededor de la caja del pez, en porcentaje del ancho y alto detectados."
        )
        self.resize_toggle.setToolTip("Redimensiona la ROI a un cuadrado fijo conservando la proporción del pez (con relleno).")
        self.resize_combo.setToolTip("Resolución de salida del redimensionado: más grande conserva detalle y cuesta más.")
        self.normalize_toggle.setToolTip("Convierte los píxeles de 0–255 a 0–1, lo que estabiliza el entrenamiento.")

        # Mejora
        self.enhance_group = self._group("Análisis y mejora", "wave")
        self.enhance_toggle = self._toggle_row("Aplicar mejora de contraste", True)
        self.enhance_group.body().addWidget(self.enhance_toggle)
        self.equal_combo = QComboBox()
        for key, label in EQUALIZATIONS.items():
            self.equal_combo.addItem(label, key)
        self.equal_combo.setCurrentIndex(1)
        self.enhance_group.body().addLayout(self._combo_row("Mejora de contraste", self.equal_combo))
        self.enhance_toggle.setToolTip("Ecualiza el histograma de cada canal RGB antes de la convolución.")
        self.equal_combo.setToolTip(
            "Sin mejora: conserva la imagen · Histograma: ecualización global por canal."
        )

        # Convolución
        self.conv_group = self._group("Filtrado y convolución", "grid")
        self.conv_toggle = self._toggle_row("Aplicar convolución", True)
        self.conv_group.body().addWidget(self.conv_toggle)
        self.kernel_combo = QComboBox()
        for key, data in KERNELS.items():
            self.kernel_combo.addItem(data["label"], key)
        self.conv_group.body().addLayout(self._combo_row("Kernel", self.kernel_combo))
        self.kernel_desc = QLabel(KERNELS["vertical"]["description"])
        self.kernel_desc.setProperty("subtle", True)
        self.kernel_desc.setWordWrap(True)
        self.conv_group.body().addWidget(self.kernel_desc)
        self.kernel_editor = KernelEditor(KERNELS["vertical"]["matrix"])
        self.conv_group.body().addWidget(self.kernel_editor)
        grid = QHBoxLayout()
        grid.setSpacing(8)
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 3)
        self.padding_spin.setValue(1)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 3)
        self.stride_spin.setValue(1)
        grid.addLayout(self._combo_row("Padding", self.padding_spin))
        grid.addLayout(self._combo_row("Stride", self.stride_spin))
        self.conv_group.body().addLayout(grid)
        self.relu_toggle = self._toggle_row("Aplicar ReLU", True)
        self.conv_group.body().addWidget(self.relu_toggle)
        self.conv_toggle.setToolTip("Activa o desactiva la etapa de convolución en el pipeline.")
        self.kernel_combo.setToolTip("Filtro 3×3 que detecta un tipo de borde o textura (Sobel, Laplaciano, relieve…).")
        self.padding_spin.setToolTip("Ceros añadidos alrededor de la imagen antes de convolucionar.")
        self.stride_spin.setToolTip("Salto del kernel al recorrer la imagen; un stride mayor reduce la salida.")
        self.relu_toggle.setToolTip("ReLU = max(0, x): elimina respuestas negativas y acelera el aprendizaje.")

        # Pooling
        self.pool_group = self._group("Pooling", "target")
        self.pool_toggle = self._toggle_row("Aplicar pooling", True)
        self.pool_group.body().addWidget(self.pool_toggle)
        self.pool_combo = QComboBox()
        for key, label in POOLINGS.items():
            self.pool_combo.addItem(label, key)
        self.pool_combo.setCurrentIndex(0)
        self.pool_group.body().addLayout(self._combo_row("Tipo", self.pool_combo))
        prow = QHBoxLayout()
        prow.setSpacing(8)
        self.pool_size = QSpinBox()
        self.pool_size.setRange(2, 4)
        self.pool_size.setValue(2)
        self.pool_stride = QSpinBox()
        self.pool_stride.setRange(1, 4)
        self.pool_stride.setValue(2)
        prow.addLayout(self._combo_row("Tamaño", self.pool_size))
        prow.addLayout(self._combo_row("Stride", self.pool_stride))
        self.pool_group.body().addLayout(prow)
        self.flatten_toggle = self._toggle_row("Aplanar (Flatten)", True)
        self.pool_group.body().addWidget(self.flatten_toggle)
        self.pool_toggle.setToolTip("Reduce la resolución del mapa de características conservando lo más importante.")
        self.pool_combo.setToolTip("Max: se queda con el máximo de cada ventana · Average: promedia la ventana.")
        self.pool_size.setToolTip("Tamaño de la ventana de pooling (2×2, 3×3 o 4×4).")
        self.pool_stride.setToolTip("Salto entre ventanas de pooling; si es igual al tamaño, no se solapan.")
        self.flatten_toggle.setToolTip("Convierte los mapas 2D en un vector 1D para la capa oculta.")

        # Acciones
        actions = Card(padding=12, spacing=8)
        self.auto_toggle = self._toggle_row("Actualización automática", True)
        self.auto_toggle.setToolTip("Recalcula el pipeline en cuanto cambia cualquier control, sin pulsar el botón.")
        actions.body().addWidget(self.auto_toggle)
        self.recalc_btn = QPushButton("Recalcular vista")
        self.recalc_btn.setToolTip("Vuelve a procesar la imagen seleccionada con la configuración actual (F5).")
        self.recalc_btn.setProperty("variant", "primary")
        self.recalc_btn.setIcon(icons.icon("refresh", ON_ACCENT, 15))
        actions.body().addWidget(self.recalc_btn)
        self.reset_btn = QPushButton("Restablecer pipeline")
        self.reset_btn.setToolTip("Devuelve todos los controles del pipeline a sus valores iniciales.")
        self.reset_btn.setProperty("variant", "ghost")
        self.reset_btn.setIcon(icons.icon("refresh", MUTED, 14))
        actions.body().addWidget(self.reset_btn)
        self.timing = QLabel("—")
        self.timing.setProperty("mono", True)
        self.timing.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions.body().addWidget(self.timing)
        layout.addWidget(actions)
        layout.addStretch(1)

        self._wire()
        self.set_config(PipelineConfig())

    # ------------------------------------------------------------------
    def _group(self, title: str, icon_name: str) -> Card:
        card = Card(padding=13, spacing=8)
        head = QHBoxLayout()
        head.setSpacing(7)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, ACCENT, 15))
        head.addWidget(ic)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {TEXT};")
        head.addWidget(lbl)
        head.addStretch(1)
        card.body().addLayout(head)
        card.body().addWidget(Divider())
        self.layout().addWidget(card)
        return card

    def _toggle_row(self, text: str, checked: bool) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        lbl = QLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 11px;")
        row.addWidget(lbl, 1)
        sw = ToggleSwitch(checked)
        row.addWidget(sw)
        w._switch = sw  # type: ignore[attr-defined]
        return w

    def _slider_row(self, text: str, lo: int, hi: int, value: int) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        label = QLabel(str(value))
        label.setFixedWidth(34)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet(f"color: {ACCENT}; font-size: 10px; font-weight: 600;")
        slider.valueChanged.connect(lambda v, l=label: l.setText(str(v)))
        return slider, label

    def _labeled_row(self, text: str, value_label: QLabel, slider: QSlider) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(7)
        lbl = QLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 10px;")
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return row

    def _combo_row(self, text: str, combo: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(3)
        lbl = QLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 10px;")
        box.addWidget(lbl)
        box.addWidget(combo)
        return box

    def _switch(self, w: QWidget) -> ToggleSwitch:
        return w._switch  # type: ignore[attr-defined]

    def _bind_stage(self, key: str, widget: QWidget) -> None:
        """Registra el interruptor propio de la tarjeta como fuente de verdad."""
        self.stage_switches[key] = self._switch(widget)

    @staticmethod
    def _sync_switch(widget: ToggleSwitch, on: bool) -> None:
        if widget.isChecked() == bool(on):
            widget.set_offset(1.0 if on else 0.0)
            return
        widget.blockSignals(True)
        widget.setChecked(bool(on))
        widget.blockSignals(False)
        widget.set_offset(1.0 if on else 0.0)

    def set_stage_enabled(self, key: str, enabled: bool) -> None:
        sw = self.stage_switches.get(key)
        if sw is not None and sw.isChecked() != bool(enabled):
            self._sync_switch(sw, enabled)

    def _set_color(self, key: str) -> None:
        self.color_buttons[key].setChecked(True)
        self._emit()

    def _wire(self) -> None:
        for key, widget in (
            ("channels", self.show_channels),
            ("conversion", self.conversion_toggle),
            ("roi", self.roi_toggle),
            ("resize", self.resize_toggle),
            ("enhance", self.enhance_toggle),
            ("normalize", self.normalize_toggle),
            ("conv", self.conv_toggle),
            ("relu", self.relu_toggle),
            ("pool", self.pool_toggle),
            ("flatten", self.flatten_toggle),
        ):
            self._bind_stage(key, widget)
        for widget in (
            self.show_channels, self.conversion_toggle, self.roi_toggle, self.resize_toggle,
            self.enhance_toggle, self.normalize_toggle, self.conv_toggle, self.relu_toggle,
            self.pool_toggle, self.flatten_toggle,
        ):
            self._switch(widget).toggled.connect(self._emit)
        self.margin_slider.valueChanged.connect(self._emit)
        self.resize_combo.currentIndexChanged.connect(self._emit)
        self.equal_combo.currentIndexChanged.connect(self._emit)
        self.pool_combo.currentIndexChanged.connect(self._emit)
        self.pool_size.valueChanged.connect(self._emit)
        self.pool_stride.valueChanged.connect(self._emit)
        self.padding_spin.valueChanged.connect(self._emit)
        self.stride_spin.valueChanged.connect(self._emit)
        self.kernel_combo.currentIndexChanged.connect(self._on_kernel_changed)
        self.kernel_editor.valueChanged.connect(self._on_kernel_edited)
        self.recalc_btn.clicked.connect(self.recalc.emit)
        self.reset_btn.clicked.connect(self._reset)

    def _on_kernel_changed(self) -> None:
        key = self.kernel_combo.currentData()
        data = KERNELS[key]
        self.kernel_editor.set_editable(key == "custom")
        if key != "custom":
            self.kernel_editor.set_values(data["matrix"])
        self.kernel_desc.setText(data["description"])
        self._emit()

    def _on_kernel_edited(self, values: list[list[float]]) -> None:
        if self.kernel_combo.currentData() == "custom":
            self._emit()

    def _reset(self) -> None:
        self.set_config(PipelineConfig())
        self._emit()

    def _emit(self) -> None:
        if not self._loading:
            self.changed.emit()

    def config(self) -> PipelineConfig:
        color = next((k for k, b in self.color_buttons.items() if b.isChecked()), "rgb")
        return PipelineConfig(
            color_space=color,
            channels_enabled=self.stage_switches["channels"].isChecked(),
            show_channels=self.stage_switches["channels"].isChecked(),
            roi_enabled=self.stage_switches["roi"].isChecked(),
            roi_margin=self.margin_slider.value() / 100.0,
            resize_enabled=self.stage_switches["resize"].isChecked(),
            resize=int(self.resize_combo.currentData()),
            conversion_enabled=self.stage_switches["conversion"].isChecked(),
            enhance_enabled=self.stage_switches["enhance"].isChecked(),
            normalize=self.stage_switches["normalize"].isChecked(),
            equalization=self.equal_combo.currentData(),
            conv_enabled=self.stage_switches["conv"].isChecked(),
            kernel_name=self.kernel_combo.currentData(),
            kernel=self.kernel_editor.values(),
            padding=self.padding_spin.value(),
            stride=self.stride_spin.value(),
            relu=self.stage_switches["relu"].isChecked(),
            pool_enabled=self.stage_switches["pool"].isChecked(),
            pooling=self.pool_combo.currentData(),
            pool_size=self.pool_size.value(),
            pool_stride=self.pool_stride.value(),
            flatten=self.stage_switches["flatten"].isChecked(),
        )

    def set_config(self, cfg: PipelineConfig) -> None:
        self._loading = True
        try:
            self.color_buttons.get(cfg.color_space, self.color_buttons["rgb"]).setChecked(True)
            for key, widget in (
                ("channels", self.show_channels),
                ("conversion", self.conversion_toggle),
                ("roi", self.roi_toggle),
                ("resize", self.resize_toggle),
                ("enhance", self.enhance_toggle),
                ("normalize", self.normalize_toggle),
                ("conv", self.conv_toggle),
                ("relu", self.relu_toggle),
                ("pool", self.pool_toggle),
                ("flatten", self.flatten_toggle),
            ):
                enabled = cfg.stage_enabled(key)
                self._sync_switch(self._switch(widget), enabled)
            self.margin_slider.setValue(int(cfg.roi_margin * 100))
            idx = self.resize_combo.findData(cfg.resize)
            if idx >= 0:
                self.resize_combo.setCurrentIndex(idx)
            idx = self.equal_combo.findData(cfg.equalization)
            if idx >= 0:
                self.equal_combo.setCurrentIndex(idx)
            idx = self.kernel_combo.findData(cfg.kernel_name)
            if idx >= 0:
                self.kernel_combo.setCurrentIndex(idx)
            self.kernel_editor.set_values(cfg.kernel)
            self.kernel_editor.set_editable(cfg.kernel_name == "custom")
            self.padding_spin.setValue(cfg.padding)
            self.stride_spin.setValue(cfg.stride)
            idx = self.pool_combo.findData(cfg.pooling)
            if idx >= 0:
                self.pool_combo.setCurrentIndex(idx)
            self.pool_size.setValue(cfg.pool_size)
            self.pool_stride.setValue(cfg.pool_stride)
        finally:
            self._loading = False


class ExplorerPage(QWidget):
    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent)
        self.main = main
        self.cfg = PipelineConfig()
        self.cfg.net_input_size = int(getattr(main.training_cfg, "input_size", 48))
        self._info: DatasetInfo | None = None
        self._thumbs: dict[str, str] = {}
        self._current: str | None = None
        self._token = 0
        self._result = None
        self._workers: list[PipelineWorker] = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._run_pipeline)

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ---------------- Lista de muestras ----------------
        left = Card(padding=12, spacing=8)
        left.setMinimumWidth(236)
        left.setMaximumWidth(300)
        head = SectionHeader("Muestras", "Selecciona una imagen")
        left.body().addWidget(head)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Buscar por nombre…")
        self.search.addAction(icons.icon("search", MUTED_2, 15), QLineEdit.ActionPosition.LeadingPosition)
        left.body().addWidget(self.search)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Todas las clases", "all")
        for cls in CLASSES:
            self.filter_combo.addItem(CLASS_LABELS[cls], cls)
        left.body().addWidget(self.filter_combo)
        self.list = QListWidget()
        self.list.setIconSize(QSize(56, 42))
        self.list.setSpacing(1)
        left.body().addWidget(self.list, 1)
        self.list_count = QLabel("0 imágenes")
        self.list_count.setProperty("subtle", True)
        left.body().addWidget(self.list_count)
        splitter.addWidget(left)

        # ---------------- Etapas ----------------
        center = QWidget()
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(0, 0, 0, 0)
        center_l.setSpacing(10)
        header = Card(padding=14, spacing=6)
        hrow = QHBoxLayout()
        hcol = QVBoxLayout()
        hcol.setSpacing(2)
        self.title = ElidedLabel("Pipeline visual")
        self.title.setProperty("h2", True)
        hcol.addWidget(self.title)
        self.subtitle = ElidedLabel("Selecciona una imagen del dataset para comenzar")
        self.subtitle.setProperty("muted", True)
        hcol.addWidget(self.subtitle)
        hrow.addLayout(hcol, 1)
        self.class_badge = Badge("—", ACCENT)
        hrow.addWidget(self.class_badge)
        self.time_badge = Badge("—", PURPLE)
        hrow.addWidget(self.time_badge)
        header.body().addLayout(hrow)
        center_l.addWidget(header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        stage_host = QWidget()
        stage_host.setObjectName("Page")
        self.stage_grid = ResponsiveGrid(spacing=12, min_item_width=235)
        sh = QVBoxLayout(stage_host)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.stage_grid)
        sh.addStretch(1)
        scroll.setWidget(stage_host)
        center_l.addWidget(scroll, 1)

        # Stage cards: una tarjeta por cada paso real del pipeline
        self.card_original = StageCard(1, "Imagen original", "Adquisición · RGB H×W×3", stage_key="read", can_toggle=False)
        self.card_channels = StageCard(2, "Espacios de color", "Canales R G B · H S V", stage_key="channels", can_toggle=True)
        self.card_roi = StageCard(3, "Segmentación Otsu", "Contorno mayor → ROI", stage_key="roi", can_toggle=True)
        self.card_resize = StageCard(4, "Redimensionado", "Escala uniforme", stage_key="resize", can_toggle=True)
        self.card_convert = StageCard(5, "Conversión de color", "Gris / HSV / RGB", stage_key="conversion", can_toggle=True)
        self.card_hist = StageCard(6, "Mejora e histograma", "Ecualización de histograma", stage_key="enhance", can_toggle=True)
        self.card_norm = StageCard(7, "Normalización", "uint8 → float32", stage_key="normalize", can_toggle=True)
        self.card_conv = StageCard(8, "Convolución", "Kernel 3×3 + padding + stride", stage_key="conv", can_toggle=True)
        self.card_relu = StageCard(9, "Activación ReLU", "max(0, x)", stage_key="relu", can_toggle=True)
        self.card_pool = StageCard(10, "Pooling", "Reducción de la matriz", stage_key="pool", can_toggle=True)
        self.card_flat = StageCard(11, "Flatten", "Vector de características", stage_key="flatten", can_toggle=True)
        self.card_net = StageCard(12, "Entrada a la red", "Tensor de la CNN", stage_key="net", can_toggle=False)
        self.hist_canvas = HistogramCanvas()
        self.card_hist.set_canvas(self.hist_canvas, "Histograma de 256 bins")
        for card in (
            self.card_original, self.card_channels, self.card_roi, self.card_resize,
            self.card_convert, self.card_hist, self.card_norm, self.card_conv,
            self.card_relu, self.card_pool, self.card_flat, self.card_net,
        ):
            self.stage_grid.add(card)
        self.stage_cards = {
            "read": self.card_original,
            "channels": self.card_channels,
            "roi": self.card_roi,
            "resize": self.card_resize,
            "conversion": self.card_convert,
            "enhance": self.card_hist,
            "normalize": self.card_norm,
            "conv": self.card_conv,
            "relu": self.card_relu,
            "pool": self.card_pool,
            "flatten": self.card_flat,
            "net": self.card_net,
        }
        for key, card in self.stage_cards.items():
            if card.can_toggle:
                card.toggled.connect(lambda on, k=key: self._on_card_toggled(k, on))
        splitter.addWidget(center)

        # ---------------- Controles ----------------
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(310)
        right_scroll.setMaximumWidth(350)
        self.controls = ControlsPanel()
        self.controls.set_config(getattr(main, "pipeline_cfg", PipelineConfig()))
        self.cfg = self.controls.config()
        self.cfg.net_input_size = int(getattr(main.training_cfg, "input_size", 48))
        for key, card in self.stage_cards.items():
            card.set_enabled(self.cfg.stage_enabled(key))
        right_scroll.setWidget(self.controls)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 900, 330])

        self.search.textChanged.connect(self._filter)
        self.filter_combo.currentIndexChanged.connect(self._filter)
        self.list.currentItemChanged.connect(self._on_item)
        self.controls.changed.connect(self._on_controls)
        self.controls.recalc.connect(self._run_pipeline)
        self.controls.auto_toggle._switch.toggled.connect(self._on_auto)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def refresh(self, info: DatasetInfo, thumbs: dict[str, str]) -> None:
        self._info = info
        self._thumbs = thumbs or {}
        current = self._current
        self._populate()
        if current:
            self.set_sample(current)
        elif info.samples:
            self.set_sample(str(info.samples[0].path))

    def _populate(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        if not self._info:
            self.list.blockSignals(False)
            return
        query = self.search.text().strip().lower()
        cls_filter = self.filter_combo.currentData()
        count = 0
        for sample in self._info.samples:
            if cls_filter != "all" and sample.class_name != cls_filter:
                continue
            if query and query not in sample.name.lower() and query not in CLASS_LABELS[sample.class_name].lower():
                continue
            item = QListWidgetItem(f"{CLASS_LABELS[sample.class_name]}\n{sample.name[:26]}")
            item.setData(Qt.ItemDataRole.UserRole, str(sample.path))
            item.setToolTip(str(sample.path))
            tp = self._thumbs.get(str(sample.path))
            if tp:
                item.setIcon(QIcon(tp))
            color = CLASS_COLORS[sample.class_name]
            item.setForeground(QColor(TEXT))
            self.list.addItem(item)
            count += 1
        self.list.blockSignals(False)
        self.list_count.setText(f"{count} imágenes")

    def _filter(self) -> None:
        current = self._current
        self._populate()
        # Restaurar selección si sigue visible
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == current:
                self.list.setCurrentRow(i)
                break

    def _on_item(self, current: QListWidgetItem | None, _prev=None) -> None:
        if not current:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        if path:
            self.set_sample(path)

    def set_sample(self, path: str) -> None:
        self._current = str(path)
        sample = next((s for s in (self._info.samples if self._info else []) if str(s.path) == str(path)), None)
        if sample:
            self.title.setText(sample.name)
            self.subtitle.setText(str(sample.path))
            self.class_badge.setText(CLASS_LABELS[sample.class_name])
            color = CLASS_COLORS[sample.class_name]
            self.class_badge.setStyleSheet(
                f"background: {rgba(color, 0.09)}; color: {color}; border: 1px solid {rgba(color, 0.33)};"
                "border-radius: 10px; padding: 2px 9px; font-size: 11px; font-weight: 600;"
            )
            # seleccionar en la lista si no está seleccionado
            for i in range(self.list.count()):
                if self.list.item(i).data(Qt.ItemDataRole.UserRole) == self._current:
                    if self.list.currentRow() != i:
                        self.list.blockSignals(True)
                        self.list.setCurrentRow(i)
                        self.list.blockSignals(False)
                    break
        self._run_pipeline()

    def _on_card_toggled(self, key: str, on: bool) -> None:
        self._token += 1
        card = self.stage_cards.get(key)
        if card:
            card.set_enabled(on)
        self.controls.set_stage_enabled(key, on)
        self.cfg = self.controls.config()
        self.cfg.net_input_size = int(self.main.training_cfg.input_size)
        self.main.pipeline_cfg = self.cfg.copy()
        self.main.pipeline_changed.emit()
        self._update_labels()
        self._run_pipeline()

    def _on_controls(self) -> None:
        self._token += 1
        self.cfg = self.controls.config()
        self.cfg.net_input_size = int(self.main.training_cfg.input_size)
        self.main.pipeline_cfg = self.cfg.copy()
        self.main.pipeline_changed.emit()
        for key, card in self.stage_cards.items():
            if card.can_toggle:
                card.set_enabled(self.cfg.stage_enabled(key))
        if self.controls.auto_toggle._switch.isChecked():  # type: ignore[attr-defined]
            self._timer.start()
        self._update_labels()

    def _on_auto(self, on: bool) -> None:
        self.controls.recalc_btn.setEnabled(not on)

    def _update_labels(self) -> None:
        self.card_roi.setToolTip(f"Margen de caja {self.cfg.roi_margin * 100:.0f} % · Otsu + contorno mayor")
        self.card_resize.setToolTip(f"Redimensionado a {self.cfg.resize}×{self.cfg.resize}")
        self.card_pool.setToolTip(f"{self.cfg.pooling} {self.cfg.pool_size}×{self.cfg.pool_size}")
        self.card_net.setToolTip(f"Tensor 3 × {self.cfg.net_input_size} × {self.cfg.net_input_size} RGB para la CNN")

    def _run_pipeline(self) -> None:
        if not self._current:
            return
        self._token += 1
        self.cfg.net_input_size = int(self.main.training_cfg.input_size)
        self.controls.timing.setText("Procesando…")
        worker = PipelineWorker(self._current, self.cfg.copy(), self._token)
        self._workers.append(worker)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.result.connect(lambda _p, w=worker: self._release(w))
        worker.signals.error.connect(lambda _m, w=worker: self._release(w))
        self.main.thread_pool.start(worker)

    def _release(self, worker: PipelineWorker) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    def _on_result(self, payload) -> None:
        token, res = payload
        if token != self._token:
            return
        self._result = res
        self.time_badge.setText(f"{res.timings.get('total', 0):.0f} ms")
        by_key = {s["key"]: s for s in res.stages}
        states = res.stage_states
        for key, card in self.stage_cards.items():
            card.set_enabled(self.cfg.stage_enabled(key))

        def detail(key: str) -> str:
            stage = by_key.get(key, {})
            return f"{stage.get('shape', '—')} · {stage.get('detail', '')}"

        h, w = res.original.shape[:2]
        self.card_original.set_image(res.original, f"{w} × {h} px · 3 canales · {res.original.dtype}")
        self.card_channels.set_image(
            res.channels, "R · G · B  |  H · S · V" if states.get("channels") else detail("channels")
        )
        self.card_roi.set_image(res.roi_rgb, detail("roi"))
        self.card_resize.set_image(res.work_rgb, detail("resize"))
        self.card_convert.set_image(res.converted_display, detail("conversion"))
        self.hist_canvas.update_hist(
            res.hist_before, res.hist_after,
            EQUALIZATIONS.get(self.cfg.equalization, "") if states.get("enhance", True) else "Desactivada",
        )
        self.card_hist.footer.setText(
            f"{detail('enhance')} · media {res.processed.mean():.3f}"
        )
        self.card_norm.set_image(
            res.normalized_display,
            f"{detail('normalize')} · rango [{res.processed.min():.3f}, {res.processed.max():.3f}]",
        )
        k = res.conv_raw
        self.card_conv.set_image(
            res.conv_display,
            f"{detail('conv')} · rango [{k.min():.2f}, {k.max():.2f}]",
        )
        positive = float((res.conv_relu > 0).mean() * 100) if res.conv_relu.size else 0.0
        max_val = float(res.conv_relu.max()) if res.conv_relu.size else 0.0
        self.card_relu.set_image(
            res.relu_display,
            f"{detail('relu')} · activaciones > 0: {positive:.1f}% · max {max_val:.2f}",
        )
        p = res.pooled
        self.card_pool.set_image(res.pooled_display, detail("pool"))
        if res.flat.size:
            flat_footer = f"{detail('flatten')} · media {res.flat.mean():.3f} · std {res.flat.std():.3f}"
        else:
            flat_footer = detail("flatten")
        self.card_flat.set_image(res.flat_image, flat_footer)
        self.card_net.set_image(
            res.net_display,
            f"{detail('net')} · listo para la CNN",
        )
        self.controls.timing.setText(
            f"{res.timings.get('total', 0):.1f} ms · {sum(1 for v in states.values() if v)} etapas activas"
        )

    def _on_error(self, message: str) -> None:
        self.controls.timing.setText("Error")
        self.main.toast(f"No se pudo procesar la imagen: {message.splitlines()[0]}", "error")
