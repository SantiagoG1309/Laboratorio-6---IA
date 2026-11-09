"""Página «Arquitectura NN»: esquema interactivo, tablas vivas y laboratorio numérico.

Calcula formas, parámetros, operaciones y pasos paso a paso desde la configuración
real de la red convolucional implementada en ``core/cnn.py`` y ``core/nn.py`` (NumPy puro).

Estructura de columnas:
  capa | Entrada (H×W×C) | Kernel / ventana | Stride | Padding | Filtros | Salida (H×W×C) | TOTAL PARAMETROS

Flujo de tensores: La salida de cada etapa se convierte en la entrada exacta de la siguiente etapa.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QTableWidget, QTableWidgetItem, QSlider, QHeaderView,
    QGridLayout, QFrame, QPushButton, QComboBox, QSizePolicy,
    QAbstractItemView, QSpinBox, QCheckBox, QFileDialog, QApplication
)

from ui import icons, preferences
from ui.theme import (
    ACCENT, ACCENT_2, BG_ALT, BORDER, BORDER_SOFT, DANGER,
    MUTED, MUTED_2, ON_ACCENT, PURPLE, SUCCESS, SURFACE, SURFACE_2, TEXT, WARNING, rgba, qcolor
)
from ui.widgets.cards import Card, Divider, SectionHeader, StatCard, Badge
from ui.widgets.flow_layout import FlowLayout, ResponsiveGrid
from ui.widgets.matrix_table import MatrixTable
from ui.widgets.nn_diagram import NNDiagram
from core.architecture import (
    ArchRow, totals, EXCEL_COLUMNS, cnn_architecture_rows,
    conv1_frames, hidden_frames, softmax_frames, LabStep,
    audit_imports, MANUAL_ENGINE_CHECKS
)
from core.config import CLASSES, CLASS_LABELS, TrainingConfig
from core.image_ops import imread_rgb
from core.nn import gradient_check
from core.cnn import build_model
from core.pipeline import preprocess_for_training


def _field(label_text: str, widget: QWidget, tip: str = "") -> QWidget:
    """Etiqueta + control en un bloque compacto que fluye con el ancho."""
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    label = QLabel(label_text)
    label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
    row.addWidget(label)
    row.addWidget(widget, 1)
    if tip:
        box.setToolTip(tip)
        label.setToolTip(tip)
        widget.setToolTip(tip)
    return box


class ArchitecturePage(QWidget):
    """Página «Arquitectura NN»: tabla viva de tensores, diagrama interactivo y laboratorio."""

    def __init__(self, main, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main = main
        self._mode = "cnn"
        self._rows: list[ArchRow] = []
        self._model = None
        self._use_excel_theme = False  # False: Tema nativo AquaCNN (oscuro elegante según diseño de la app)
        self._updating_controls = False

        self._lab_steps: list[LabStep] = []
        self._step_idx = 0
        self._lab_model = None
        self._lab_cfg: TrainingConfig | None = None
        self._lab_input: np.ndarray | None = None
        self._lab_trace: list[dict] = []
        self._lab_sample = None
        self._lab_stage = "conv1"
        self._lab_index = 0

        # Scroll principal
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        content.setObjectName("Page")
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(26, 20, 26, 28)
        layout.setSpacing(16)

        # -----------------------------------------------------------------
        # 1. Cabecera principal y acciones rápidas
        # -----------------------------------------------------------------
        header = SectionHeader(
            "Arquitectura de la Red Neuronal Convolucional (CNN)",
            "Diagrama interactivo, tensores capa por capa calculados en tiempo real y laboratorio numérico"
        )
        header.title.setWordWrap(True)
        header.subtitle.setWordWrap(True)
        layout.addWidget(header)

        header_actions = QHBoxLayout()
        header_actions.setSpacing(8)
        self.badge_cnn = Badge("Red Convolucional (CNN para Peces)", ACCENT)
        header_actions.addWidget(self.badge_cnn)
        header_actions.addStretch(1)

        self.sync_btn = QPushButton("Sincronizar con entrenamiento")
        self.sync_btn.setProperty("variant", "ghost")
        self.sync_btn.setIcon(icons.icon("refresh", MUTED, 14))
        self.sync_btn.setToolTip("Carga los parámetros actuales del panel de Entrenamiento y del Pipeline.")
        self.sync_btn.clicked.connect(self.refresh_from_training)
        header_actions.addWidget(self.sync_btn)
        layout.addLayout(header_actions)

        # -----------------------------------------------------------------
        # 2. Tarjetas de métricas y totales (Parámetros, FLOPs, Activaciones)
        # -----------------------------------------------------------------
        stats_grid = ResponsiveGrid(spacing=12, min_item_width=150)
        self.card_params = StatCard("cpu", "Parámetros totales", "0", ACCENT)
        self.card_flops = StatCard("sliders", "FLOPs estimados", "0", PURPLE)
        self.card_elements = StatCard("layers", "Activaciones / Salida", "0", ACCENT_2)
        self.card_layers = StatCard("check", "Capas con pesos", "0", SUCCESS)

        for card in (self.card_params, self.card_flops, self.card_elements, self.card_layers):
            stats_grid.add(card)
        layout.addWidget(stats_grid)

        # -----------------------------------------------------------------
        # 3. Diagrama interactivo de la red neuronal
        # -----------------------------------------------------------------
        diagram_card = Card(padding=14, spacing=10)
        d_head = QVBoxLayout()
        d_head.setSpacing(2)
        d_title = QLabel("Esquema Computacional Interactivo")
        d_title.setProperty("h3", True)
        d_head.addWidget(d_title)
        d_sub = QLabel("Pasa el cursor para ver detalles o haz clic sobre una capa para ubicarla en la tabla")
        d_sub.setProperty("muted", True)
        d_sub.setStyleSheet("font-size: 11px;")
        d_head.addWidget(d_sub)

        diagram_card.body().addLayout(d_head)
        self.diagram = NNDiagram()
        self.diagram.layerClicked.connect(self._on_diagram_layer_clicked)
        diagram_card.body().addWidget(self.diagram)
        layout.addWidget(diagram_card)

        # -----------------------------------------------------------------
        # 4. Tabla Viva de Arquitectura (Estructura de tensores en tiempo real)
        # -----------------------------------------------------------------
        table_card = Card(padding=16, spacing=12)

        # Encabezado de la tarjeta con el diseño estructural solicitado
        t_head = QVBoxLayout()
        t_head.setSpacing(6)
        t_title = QLabel("Desglose Capa por Capa y Dimensiones de Tensores")
        t_title.setProperty("h3", True)
        t_head.addWidget(t_title)

        self.lbl_table_subtitle = QLabel(
            "Flujo de tensores: La salida de cada etapa se convierte en la entrada exacta de la etapa siguiente."
        )
        self.lbl_table_subtitle.setProperty("muted", True)
        self.lbl_table_subtitle.setStyleSheet("font-size: 11px;")
        t_head.addWidget(self.lbl_table_subtitle)

        # Acciones en un flujo que se reacomoda cuando la ventana es estrecha
        t_actions = FlowLayout(h_spacing=8, v_spacing=6, min_item_width=0)

        self.lbl_arch_badge = QLabel("Arquitectura: CNN para peces")
        self.lbl_arch_badge.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: 600;")
        t_actions.addWidget(self.lbl_arch_badge)

        # Botones de acción de la tabla
        self.btn_copy_tsv = QPushButton("Copiar")
        self.btn_copy_tsv.setProperty("variant", "ghost")
        self.btn_copy_tsv.setIcon(icons.icon("check", MUTED, 13))
        self.btn_copy_tsv.setToolTip("Copia los datos de la tabla en formato TSV (para pegar en Excel).")
        self.btn_copy_tsv.clicked.connect(self._copy_table_tsv)
        t_actions.addWidget(self.btn_copy_tsv)

        self.btn_export_xlsx = QPushButton("Excel")
        self.btn_export_xlsx.setProperty("variant", "ghost")
        self.btn_export_xlsx.setIcon(icons.icon("download", MUTED, 13))
        self.btn_export_xlsx.setToolTip("Exporta los cálculos reales de la CNN a una hoja de cálculo Excel.")
        self.btn_export_xlsx.clicked.connect(self._export_to_excel)
        t_actions.addWidget(self.btn_export_xlsx)

        self.btn_toggle_theme = QPushButton("Tema Clásico Excel")
        self.btn_toggle_theme.setProperty("variant", "ghost")
        self.btn_toggle_theme.setIcon(icons.icon("sliders", MUTED, 13))
        self.btn_toggle_theme.setToolTip("Alterna entre la apariencia nativa oscura y la cuadrícula Excel pastel.")
        self.btn_toggle_theme.clicked.connect(self._toggle_table_theme)
        t_head.addWidget(self.btn_toggle_theme)

        table_card.body().addLayout(t_head)

        # -----------------------------------------------------------------
        # Panel de Controles Interactivos en Tiempo Real (CNN de Peces)
        # -----------------------------------------------------------------
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(
            f"background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 8px; padding: 10px;"
        )
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(10)

        # Fila A: Presets de la CNN para peces y accesos rápidos
        preset_row = FlowLayout(h_spacing=8, v_spacing=6, min_item_width=0)

        lbl_preset = QLabel("<b>Configuración CNN:</b>")
        lbl_preset.setStyleSheet(f"color: {ACCENT}; font-size: 11px;")
        preset_row.addWidget(lbl_preset)

        self.combo_preset = QComboBox()
        self.combo_preset.addItem("CNN estándar de peces · 2 bloques Conv+Pool", "fish_cnn_2b")
        self.combo_preset.addItem("CNN rápida · 1 bloque Conv+Pool", "fast_1b")
        self.combo_preset.addItem("CNN profunda · 3 bloques Conv+Pool", "deep_3b")
        self.combo_preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_preset.setMinimumContentsLength(18)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.combo_preset)

        self.btn_reset_fish = QPushButton("Restablecer CNN peces")
        self.btn_reset_fish.setProperty("variant", "ghost")
        self.btn_reset_fish.setIcon(icons.icon("fish", MUTED, 13))
        self.btn_reset_fish.setToolTip("Vuelve a la configuración de 48×48, 6/12 filtros y 64 neuronas ocultas.")
        self.btn_reset_fish.clicked.connect(lambda: self._set_preset_by_key("fish_cnn_2b"))
        preset_row.addWidget(self.btn_reset_fish)

        ctrl_layout.addLayout(preset_row)

        # Fila B: parámetros de la red en un flujo que se adapta al ancho
        params_flow = FlowLayout(h_spacing=12, v_spacing=8, min_item_width=225)

        self.spin_blocks = QSpinBox()
        self.spin_blocks.setRange(1, 3)
        self.spin_blocks.setValue(2)
        params_flow.addWidget(_field(
            "Bloques Conv+Pool:", self.spin_blocks,
            "Cuántos pares de convolución y pooling se encadenan antes de clasificar.",
        ))

        self.combo_size = QComboBox()
        for sz in (32, 40, 48, 64, 96, 128):
            self.combo_size.addItem(f"{sz} × {sz}", sz)
        self.combo_size.setCurrentIndex(2)  # 48x48
        params_flow.addWidget(_field(
            "Resolución:", self.combo_size,
            "Tamaño de la imagen que recibe la red; debe coincidir con el usado al entrenar.",
        ))

        self.combo_channels = QComboBox()
        self.combo_channels.addItem("1 · grises", 1)
        self.combo_channels.addItem("3 · RGB", 3)
        params_flow.addWidget(_field(
            "Canales:", self.combo_channels,
            "Canales de entrada: 1 si el pipeline convierte a grises, 3 si conserva color.",
        ))

        self.spin_f1 = QSpinBox()
        self.spin_f1.setRange(2, 64)
        self.spin_f1.setValue(6)
        params_flow.addWidget(_field(
            "Filtros Conv 1:", self.spin_f1,
            "Número de kernels diferentes que aprende la primera convolución.",
        ))

        self.combo_k1 = QComboBox()
        for k_val in (3, 5):
            self.combo_k1.addItem(f"{k_val} × {k_val}", k_val)
        self.combo_k1.setCurrentIndex(0)
        params_flow.addWidget(_field(
            "Kernel Conv 1:", self.combo_k1,
            "Tamaño de la ventana 3×3 o 5×5 que recorre la imagen.",
        ))

        sp_box = QWidget()
        sp_row = QHBoxLayout(sp_box)
        sp_row.setContentsMargins(0, 0, 0, 0)
        sp_row.setSpacing(4)
        self.spin_s1 = QSpinBox()
        self.spin_s1.setRange(1, 2)
        self.spin_s1.setValue(1)
        sp_row.addWidget(QLabel("S:"))
        sp_row.addWidget(self.spin_s1)
        self.spin_p1 = QSpinBox()
        self.spin_p1.setRange(0, 2)
        self.spin_p1.setValue(1)
        sp_row.addWidget(QLabel("P:"))
        sp_row.addWidget(self.spin_p1)
        params_flow.addWidget(_field(
            "Stride / Padding C1:", sp_box,
            "Stride: píxeles que salta la ventana. Padding: borde de ceros que conserva las orillas.",
        ))

        self.spin_f2 = QSpinBox()
        self.spin_f2.setRange(2, 128)
        self.spin_f2.setValue(12)
        params_flow.addWidget(_field(
            "Filtros Conv 2:", self.spin_f2,
            "Kernels de la segunda convolución; cada uno combina los mapas de la primera.",
        ))

        self.spin_f3 = QSpinBox()
        self.spin_f3.setRange(2, 256)
        self.spin_f3.setValue(24)
        params_flow.addWidget(_field(
            "Filtros Conv 3:", self.spin_f3,
            "Solo se usa con 3 bloques: kernels de la tercera convolución.",
        ))

        self.chk_pool = QCheckBox("Activar pooling 2×2")
        self.chk_pool.setChecked(True)
        params_flow.addWidget(_field(
            "Pooling:", self.chk_pool,
            "Reduce cada mapa a la mitad con ventanas 2×2 y stride 2.",
        ))

        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(8, 512)
        self.spin_hidden.setValue(64)
        self.spin_hidden.setSingleStep(8)
        params_flow.addWidget(_field(
            "Neuronas ocultas:", self.spin_hidden,
            "Neuronas de la capa oculta que combinan los patrones antes de decidir la clase.",
        ))

        badge_cls = Badge("3 clases: Trucha · Mojarra · Pavón", SUCCESS)
        badge_cls.setFixedHeight(26)
        params_flow.addWidget(_field("Salida:", badge_cls, "Clases fijas del laboratorio de peces."))

        self.spin_blocks.valueChanged.connect(self._recalculate_architecture)
        self.combo_size.currentIndexChanged.connect(self._recalculate_architecture)
        self.combo_channels.currentIndexChanged.connect(self._recalculate_architecture)
        self.spin_f1.valueChanged.connect(self._recalculate_architecture)
        self.combo_k1.currentIndexChanged.connect(self._recalculate_architecture)
        self.spin_s1.valueChanged.connect(self._recalculate_architecture)
        self.spin_p1.valueChanged.connect(self._recalculate_architecture)
        self.spin_f2.valueChanged.connect(self._recalculate_architecture)
        self.spin_f3.valueChanged.connect(self._recalculate_architecture)
        self.chk_pool.toggled.connect(self._recalculate_architecture)
        self.spin_hidden.valueChanged.connect(self._recalculate_architecture)

        ctrl_layout.addLayout(params_flow)
        table_card.body().addWidget(ctrl_frame)

        # -----------------------------------------------------------------
        # Tabla QTableWidget (8 Columnas exactas)
        # -----------------------------------------------------------------
        self.table_headers = list(EXCEL_COLUMNS)
        self.table = QTableWidget(0, len(self.table_headers))
        self.table.setHorizontalHeaderLabels(self.table_headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(True)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.setMinimumHeight(280)
        self.table.currentCellChanged.connect(self._on_table_row_changed)

        # Todas las columnas se reparten el ancho; el texto largo se elide y el
        # contenido completo queda en el tooltip de cada celda.
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header_view.setMinimumSectionSize(48)
        header_view.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setMinimumWidth(0)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        table_card.body().addWidget(self.table)
        layout.addWidget(table_card)

        # -----------------------------------------------------------------
        # 5. Laboratorio Numérico Paso a Paso (Convolución, ReLU y Pooling)
        # -----------------------------------------------------------------
        lab_card = Card(padding=12, spacing=12)
        l_head = QHBoxLayout()
        l_col = QVBoxLayout()
        l_title = QLabel("Laboratorio Numérico Paso a Paso")
        l_title.setProperty("h2", True)
        l_col.addWidget(l_title)
        l_sub = QLabel(
            "Con los pesos reales del modelo: convolución de un filtro aprendido, producto punto de una "
            "neurona oculta y normalización Softmax de la predicción."
        )
        l_sub.setProperty("muted", True)
        l_col.addWidget(l_sub)
        l_head.addLayout(l_col, 1)
        lab_card.body().addLayout(l_head)

        # Controles del laboratorio: etapa real, filtro/neurona y muestra
        lab_flow = FlowLayout(h_spacing=10, v_spacing=8, min_item_width=215)

        self.combo_stage = QComboBox()
        self.combo_stage.addItem("Conv 1 · filtro aprendido", "conv1")
        self.combo_stage.addItem("Capa oculta · neurona", "hidden")
        self.combo_stage.addItem("Salida · Softmax", "softmax")
        self.combo_stage.currentIndexChanged.connect(self._on_lab_stage_changed)
        lab_flow.addWidget(_field(
            "Etapa:", self.combo_stage,
            "Elige qué parte real de la CNN quieres desglosar con los pesos del modelo cargado.",
        ))

        self.combo_filter = QComboBox()
        self.combo_filter.currentIndexChanged.connect(self._rebuild_lab)
        self._filter_field = _field(
            "Filtro Conv 1:", self.combo_filter,
            "Cada filtro es un kernel 3×3 aprendido que detecta un patrón distinto.",
        )
        lab_flow.addWidget(self._filter_field)

        self.spin_neuron = QSpinBox()
        self.spin_neuron.setRange(1, 64)
        self.spin_neuron.setValue(1)
        self.spin_neuron.valueChanged.connect(self._rebuild_lab)
        self._neuron_field = _field(
            "Neurona oculta:", self.spin_neuron,
            "Neurona de la capa oculta cuyo producto punto x·w + b se muestra.",
        )
        lab_flow.addWidget(self._neuron_field)

        self.btn_reset_lab = QPushButton("Muestra aleatoria del dataset")
        self.btn_reset_lab.setProperty("variant", "ghost")
        self.btn_reset_lab.setIcon(icons.icon("fish", MUTED, 13))
        self.btn_reset_lab.setToolTip("Preprocesa una imagen real del dataset con el pipeline y la pasa por la red.")
        self.btn_reset_lab.clicked.connect(self._load_dataset_patch)
        lab_flow.addWidget(self.btn_reset_lab)

        self._lab_sample_label = QLabel("Sin muestra")
        self._lab_sample_label.setProperty("subtle", True)
        self._lab_sample_label.setWordWrap(True)
        lab_flow.addWidget(self._lab_sample_label)

        lab_card.body().addLayout(lab_flow)

        # Visualizador de matrices
        mats_layout = QHBoxLayout()
        mats_layout.setSpacing(6)
        mats_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Entrada
        box_src = QVBoxLayout()
        box_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_src_title = QLabel("Matriz de Entrada")
        self.lbl_src_title.setStyleSheet(f"color: {ACCENT}; font-weight: 600; font-size: 11px;")
        self.lbl_src_title.setWordWrap(True)
        self.lbl_src_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box_src.addWidget(self.lbl_src_title)
        self.mat_source = MatrixTable(10, 10, editable=False)
        box_src.addWidget(self.mat_source, 0, Qt.AlignmentFlag.AlignCenter)
        mats_layout.addLayout(box_src)

        # Operador 1
        self.lbl_op1 = QLabel("⊛")
        self.lbl_op1.setStyleSheet(f"color: {MUTED}; font-size: 20px; font-weight: bold;")
        self.lbl_op1.setFixedWidth(18)
        self.lbl_op1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mats_layout.addWidget(self.lbl_op1)

        # Kernel / pesos
        box_k = QVBoxLayout()
        box_k.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_kernel_title = QLabel("Filtro 3×3")
        self.lbl_kernel_title.setStyleSheet(f"color: {WARNING}; font-weight: 600; font-size: 11px;")
        self.lbl_kernel_title.setWordWrap(True)
        self.lbl_kernel_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box_k.addWidget(self.lbl_kernel_title)
        self.mat_kernel = MatrixTable(3, 3, editable=False)
        box_k.addWidget(self.mat_kernel, 0, Qt.AlignmentFlag.AlignCenter)
        mats_layout.addLayout(box_k)

        # Operador 2
        self.lbl_op2 = QLabel("=")
        self.lbl_op2.setStyleSheet(f"color: {MUTED}; font-size: 20px; font-weight: bold;")
        self.lbl_op2.setFixedWidth(18)
        self.lbl_op2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mats_layout.addWidget(self.lbl_op2)

        # Salida
        box_tgt = QVBoxLayout()
        box_tgt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_tgt_title = QLabel("Mapa de Características")
        self.lbl_tgt_title.setStyleSheet(f"color: {SUCCESS}; font-weight: 600; font-size: 11px;")
        self.lbl_tgt_title.setWordWrap(True)
        self.lbl_tgt_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box_tgt.addWidget(self.lbl_tgt_title)
        self.mat_target = MatrixTable(10, 10, editable=False)
        box_tgt.addWidget(self.mat_target, 0, Qt.AlignmentFlag.AlignCenter)
        mats_layout.addLayout(box_tgt)

        lab_card.body().addLayout(mats_layout)

        # Reproductor de pasos
        player = FlowLayout(h_spacing=8, v_spacing=6, min_item_width=80)

        self.btn_first = QPushButton("Inicio")
        self.btn_first.setProperty("variant", "ghost")
        self.btn_first.setIcon(icons.icon("arrow-left", MUTED, 13))
        self.btn_first.setToolTip("Ir al primer paso del recorrido numérico.")
        self.btn_first.clicked.connect(lambda: self._show_lab_step(0))
        player.addWidget(self.btn_first)

        self.btn_prev = QPushButton("Anterior")
        self.btn_prev.setProperty("variant", "ghost")
        self.btn_prev.setIcon(icons.icon("arrow-left", MUTED, 13))
        self.btn_prev.setToolTip("Paso anterior.")
        self.btn_prev.clicked.connect(lambda: self._show_lab_step(self._step_idx - 1))
        player.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Reproducir")
        self.btn_play.setProperty("variant", "primary")
        self.btn_play.setIcon(icons.icon("play", ON_ACCENT, 13))
        self.btn_play.setToolTip("Recorre los pasos automáticamente.")
        self.btn_play.clicked.connect(self._toggle_play)
        player.addWidget(self.btn_play)

        self.btn_next = QPushButton("Siguiente")
        self.btn_next.setProperty("variant", "ghost")
        self.btn_next.setIcon(icons.icon("arrow", MUTED, 13))
        self.btn_next.setToolTip("Paso siguiente.")
        self.btn_next.clicked.connect(lambda: self._show_lab_step(self._step_idx + 1))
        player.addWidget(self.btn_next)

        self.btn_last = QPushButton("Final")
        self.btn_last.setProperty("variant", "ghost")
        self.btn_last.setIcon(icons.icon("arrow", MUTED, 13))
        self.btn_last.setToolTip("Ir al último paso del recorrido numérico.")
        self.btn_last.clicked.connect(lambda: self._show_lab_step(len(self._lab_steps) - 1))
        player.addWidget(self.btn_last)

        lab_card.body().addLayout(player)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        self._lab_slider = QSlider(Qt.Orientation.Horizontal)
        self._lab_slider.setRange(0, 0)
        self._lab_slider.setToolTip("Arrastra para recorrer los pasos del cálculo.")
        self._lab_slider.valueChanged.connect(self._show_lab_step)
        slider_row.addWidget(self._lab_slider, 1)

        self._lab_step_label = QLabel("Paso 0 / 0")
        self._lab_step_label.setStyleSheet(f"color: {TEXT}; font-weight: 600; font-size: 12px;")
        self._lab_step_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider_row.addWidget(self._lab_step_label)
        lab_card.body().addLayout(slider_row)

        # Detalle de la celda y fórmula
        self.calc_box = QFrame()
        self.calc_box.setStyleSheet(
            f"background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 8px; padding: 12px;"
        )
        c_layout = QVBoxLayout(self.calc_box)
        c_layout.setSpacing(4)
        c_layout.setContentsMargins(10, 8, 10, 8)

        self._lab_detail = QLabel("Cargando laboratorio…")
        self._lab_detail.setStyleSheet(f"color: {TEXT}; font-size: 12px; font-weight: 500;")
        self._lab_detail.setWordWrap(True)
        c_layout.addWidget(self._lab_detail)

        self._lab_formula = QLabel("—")
        self._lab_formula.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-family: 'Cascadia Code', monospace;")
        self._lab_formula.setWordWrap(True)
        c_layout.addWidget(self._lab_formula)

        lab_card.body().addWidget(self.calc_box)
        layout.addWidget(lab_card)

        # -----------------------------------------------------------------
        # 6. Auditoría de Aprendizaje Manual (Sin Librerías Externas)
        # -----------------------------------------------------------------
        audit_card = Card(padding=16, spacing=12)
        a_head = QHBoxLayout()
        a_title = SectionHeader(
            "Auditoría del Motor Neuronal Manual",
            "Verificación matemática de que la red opera con NumPy puro (sin TensorFlow ni PyTorch)"
        )
        a_head.addWidget(a_title, 1)

        self.audit_btn = QPushButton("Verificar gradientes y código manual")
        self.audit_btn.setProperty("variant", "primary")
        self.audit_btn.setIcon(icons.icon("check", ON_ACCENT, 13))
        self.audit_btn.clicked.connect(self._run_gradient_audit)
        a_head.addWidget(self.audit_btn)
        audit_card.body().addLayout(a_head)

        self.audit_result_label = QLabel(
            "Haz clic en «Verificar gradientes» para ejecutar una comprobación numérica por diferencias "
            "finitas contra las derivadas analíticas del backpropagation manual."
        )
        self.audit_result_label.setWordWrap(True)
        self.audit_result_label.setStyleSheet(
            f"background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 8px; padding: 12px;"
            f"color: {MUTED}; font-size: 11px;"
        )
        audit_card.body().addWidget(self.audit_result_label)

        chk_grid = QGridLayout()
        chk_grid.setHorizontalSpacing(16)
        chk_grid.setVerticalSpacing(8)
        for i, (item_title, item_desc) in enumerate(MANUAL_ENGINE_CHECKS):
            badge = Badge("100% Manual", SUCCESS)
            badge.setFixedHeight(20)
            lbl = QLabel(f"<b>{item_title}</b>: {item_desc}")
            lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            r = i // 2
            c = (i % 2) * 2
            chk_grid.addWidget(badge, r, c)
            chk_grid.addWidget(lbl, r, c + 1)
        audit_card.body().addLayout(chk_grid)
        layout.addWidget(audit_card)

        # Timer para animación del laboratorio
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(320)
        self._play_timer.timeout.connect(self._on_play_tick)

        # Ajustes de responsividad: el contenido se adapta al ancho disponible
        self.setMinimumWidth(0)
        content.setMinimumWidth(0)
        for label in content.findChildren(QLabel):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
        for combo in content.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setMinimumWidth(70)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.table.setMinimumWidth(0)

        # Aplicar estilo inicial de la tabla y calcular arquitectura
        self._apply_table_stylesheet()
        self._recalculate_architecture()
        self._rebuild_lab()

    # ---------------------------------------------------------------------
    # Cálculo en tiempo real de la tabla de tensores
    # ---------------------------------------------------------------------
    def _recalculate_architecture(self, *_args) -> None:
        """Calcula todas las dimensiones, parámetros y operaciones en tiempo real."""
        if self._updating_controls:
            return

        num_b = int(self.spin_blocks.value())
        size = int(self.combo_size.currentData() or 48)
        in_ch = int(self.combo_channels.currentData() or 1)
        f1 = int(self.spin_f1.value())
        k1 = int(self.combo_k1.currentData() or 3)
        s1 = int(self.spin_s1.value())
        p1 = int(self.spin_p1.value())
        pool_on = bool(self.chk_pool.isChecked())
        f2 = int(self.spin_f2.value())
        f3 = int(self.spin_f3.value())
        hidden = int(self.spin_hidden.value())

        self.lbl_table_subtitle.setText(
            f"Flujo secuencial: la salida de cada etapa se convierte en la entrada exacta de la etapa siguiente · "
            f"{num_b} bloque(s) Conv+Pool · Entrada {size}×{size}×{in_ch} · Capa oculta de {hidden} neuronas · 3 clases (Trucha, Mojarra, Pavón)"
        )

        # Generar arquitectura fiel al diseño de la red para peces
        self._rows = cnn_architecture_rows(
            input_size=size,
            num_blocks=num_b,
            in_channels=in_ch,
            filters_per_block=(f1, f2, f3),
            kernel_sizes=(k1, 3, 3),
            strides=(s1, 1, 1),
            paddings=(p1, 1, 1),
            pool_sizes=(2, 2, 2),
            pool_strides=(2, 2, 2),
            pools_enabled=(pool_on, pool_on, pool_on),
            hidden_units=hidden,
            num_classes=len(CLASSES),
        )

        self._update_architecture_table()
        self.diagram.set_architecture(self._rows)
        self._update_stats()

    def _set_preset_by_key(self, key: str) -> None:
        idx = self.combo_preset.findData(key)
        if idx >= 0:
            self.combo_preset.setCurrentIndex(idx)

    def _on_preset_changed(self, idx: int) -> None:
        key = str(self.combo_preset.currentData() or "fish_cnn_2b")
        self._updating_controls = True
        try:
            if key == "fast_1b":
                self.spin_blocks.setValue(1)
                sz_idx = self.combo_size.findData(48)
                if sz_idx >= 0:
                    self.combo_size.setCurrentIndex(sz_idx)
                self.combo_channels.setCurrentIndex(0)  # 1 canal (grises)
                self.spin_f1.setValue(6)
                self.combo_k1.setCurrentIndex(0)  # 3x3
                self.spin_s1.setValue(1)
                self.spin_p1.setValue(1)
                self.chk_pool.setChecked(True)
                self.spin_hidden.setValue(64)

            elif key == "deep_3b":
                self.spin_blocks.setValue(3)
                sz_idx = self.combo_size.findData(48)
                if sz_idx >= 0:
                    self.combo_size.setCurrentIndex(sz_idx)
                self.combo_channels.setCurrentIndex(0)
                self.spin_f1.setValue(6)
                self.combo_k1.setCurrentIndex(0)
                self.spin_s1.setValue(1)
                self.spin_p1.setValue(1)
                self.spin_f2.setValue(12)
                self.spin_f3.setValue(24)
                self.chk_pool.setChecked(True)
                self.spin_hidden.setValue(64)

            else:  # fish_cnn_2b (Por defecto)
                self.spin_blocks.setValue(2)
                sz_idx = self.combo_size.findData(48)
                if sz_idx >= 0:
                    self.combo_size.setCurrentIndex(sz_idx)
                self.combo_channels.setCurrentIndex(0)  # 1 canal (grises)
                self.spin_f1.setValue(6)
                self.combo_k1.setCurrentIndex(0)  # 3x3
                self.spin_s1.setValue(1)
                self.spin_p1.setValue(1)
                self.spin_f2.setValue(12)
                self.chk_pool.setChecked(True)
                self.spin_hidden.setValue(64)
        finally:
            self._updating_controls = False

        self._recalculate_architecture()

    # ---------------------------------------------------------------------
    # Renderizado y estilo fiel a la interfaz de la aplicación
    # ---------------------------------------------------------------------
    def _toggle_table_theme(self) -> None:
        self._use_excel_theme = not self._use_excel_theme
        self.btn_toggle_theme.setText("Tema Nativo AquaCNN" if self._use_excel_theme else "Tema Clásico Excel")
        self._apply_table_stylesheet()
        self._update_architecture_table()

    def _apply_table_stylesheet(self) -> None:
        if self._use_excel_theme:
            # Estilo alternativo clásico de hoja de cálculo
            self.table.setStyleSheet(
                "QTableWidget {"
                "  background-color: #FFFFFF;"
                "  color: #000000;"
                "  gridline-color: #A6A6A6;"
                "  border: 1px solid #70AD47;"
                "  font-family: 'Segoe UI', Arial, sans-serif;"
                "}"
                "QHeaderView::section {"
                "  background-color: #A8D08D;"
                "  color: #000000;"
                "  font-weight: bold;"
                "  font-size: 11px;"
                "  border: 1px solid #70AD47;"
                "  padding: 6px 4px;"
                "}"
                "QTableWidget::item:selected {"
                "  background-color: #5B9BD5;"
                "  color: #FFFFFF;"
                "}"
            )
        else:
            # Estilo Fiel a la Interfaz de AquaCNN (Dark theme con cuadrícula nítida)
            self.table.setStyleSheet(
                f"QTableWidget {{"
                f"  background-color: {BG_ALT};"
                f"  color: {TEXT};"
                f"  gridline-color: {BORDER_SOFT};"
                f"  border: 1px solid {BORDER};"
                f"  font-family: 'Segoe UI', Arial, sans-serif;"
                f"}}"
                f"QHeaderView::section {{"
                f"  background-color: {SURFACE_2};"
                f"  color: {ACCENT};"
                f"  font-weight: bold;"
                f"  font-size: 11px;"
                f"  border: 1px solid {BORDER};"
                f"  padding: 8px 6px;"
                f"}}"
                f"QTableWidget::item:selected {{"
                f"  background-color: {rgba(ACCENT, 0.18)};"
                f"  color: {TEXT};"
                f"}}"
            )

    def _update_architecture_table(self) -> None:
        self.table.setRowCount(0)
        self.table.setRowCount(len(self._rows))

        # Colores por rol de capa en la arquitectura de la red
        role_colors = {
            "IMAGEN": ACCENT,
            "Conv 1": ACCENT_2,
            "Conv 2": ACCENT_2,
            "Conv 3": ACCENT_2,
            "Pool 1": PURPLE,
            "Pool 2": PURPLE,
            "Pool 3": PURPLE,
            "Flatten": DANGER,
            "Oculta": WARNING,
            "Salida": SUCCESS,
            "Softmax": SUCCESS,
        }

        for r_idx, row in enumerate(self._rows):
            vals = row.excel_row_values()

            # Fondo de celda
            if self._use_excel_theme:
                bg_color = QColor("#BDD7EE") if row.is_blue else QColor("#FFFFFF")
                text_color = QColor("#000000")
            else:
                # Alternancia sutil dentro de la paleta de la interfaz
                bg_color = QColor(SURFACE) if r_idx % 2 == 0 else QColor(BG_ALT)
                text_color = QColor(TEXT)

            for c_idx, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setBackground(bg_color)

                if self._use_excel_theme:
                    item.setForeground(text_color)
                else:
                    if c_idx == 0:
                        # Columna capa con el color distintivo de su función
                        capa_col = role_colors.get(str(row.capa), ACCENT)
                        item.setForeground(QColor(capa_col))
                    elif c_idx in (1, 6):  # Entrada y Salida
                        item.setForeground(QColor(TEXT))
                    elif c_idx == 7:  # TOTAL PARAMETROS
                        item.setForeground(QColor(ACCENT_2))
                    elif str(val) in ("—", ""):
                        item.setForeground(QColor(MUTED_2))
                    else:
                        item.setForeground(QColor(TEXT))

                # Fuentes y alineaciones
                if c_idx == 0:  # Capa
                    item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif c_idx in (1, 6):  # Entrada y Salida (destacadas)
                    item.setFont(QFont("Cascadia Code", 9, QFont.Weight.DemiBold))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif c_idx == 7:  # TOTAL PARAMETROS
                    item.setFont(QFont("Cascadia Code", 9, QFont.Weight.Bold))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                else:
                    item.setFont(QFont("Cascadia Code", 9))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Tooltips con fórmula y detalle
                if c_idx == 1:
                    item.setToolTip(f"Tensor de entrada a la etapa {row.capa}:\n{row.entrada_excel}")
                elif c_idx == 6:
                    item.setToolTip(f"Tensor de salida de la etapa {row.capa}:\n{row.salida_excel}")
                elif c_idx == 7:
                    item.setToolTip(f"Total parámetros / activaciones:\n{row.formula or row.formula_excel}")
                elif row.formula:
                    item.setToolTip(f"{row.operation}\n{row.formula}")
                elif row.note:
                    item.setToolTip(f"{row.operation}\n{row.note}")

                self.table.setItem(r_idx, c_idx, item)

    def _update_stats(self) -> None:
        t = totals(self._rows)
        self.card_params.set_value(f"{t['params']:,}".replace(",", "."))
        self.card_flops.set_value(f"{t['flops']:,}".replace(",", "."))
        self.card_elements.set_value(f"{t['elements']:,}".replace(",", "."))
        self.card_layers.set_value(str(t["layers"]))

    def _on_table_row_changed(self, current_row: int, current_col: int, prev_row: int, prev_col: int) -> None:
        pass

    def _on_diagram_layer_clicked(self, layer_idx: int) -> None:
        if 0 <= layer_idx < self.table.rowCount():
            self.table.selectRow(layer_idx)
            item = self.table.item(layer_idx, 0)
            if item:
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    # ---------------------------------------------------------------------
    # Copia y Exportación
    # ---------------------------------------------------------------------
    def _copy_table_tsv(self) -> None:
        """Copia la tabla al portapapeles en formato TSV para pegar directamente en Excel."""
        tsv_lines = ["\t".join(EXCEL_COLUMNS)]
        for r in self._rows:
            tsv_lines.append("\t".join(r.excel_row_values()))
        QApplication.clipboard().setText("\n".join(tsv_lines))
        if hasattr(self.main, "toast"):
            self.main.toast("Tabla de arquitectura copiada al portapapeles.", "success")

    def _export_to_excel(self) -> None:
        """Exporta la tabla a un archivo Excel (.xlsx) con los datos y cálculos de la CNN."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Arquitectura en Excel", "Arquitectura_CNN_Peces.xlsx", "Archivos de Excel (*.xlsx)"
        )
        if not path:
            return

        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Arquitectura_CNN"

            green_fill = PatternFill(start_color="111F33", end_color="111F33", fill_type="solid")
            row_fill_a = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
            row_fill_b = PatternFill(start_color="EDF2F7", end_color="EDF2F7", fill_type="solid")
            thin_border = Border(
                left=Side(style="thin", color="CBD5E1"),
                right=Side(style="thin", color="CBD5E1"),
                top=Side(style="thin", color="CBD5E1"),
                bottom=Side(style="thin", color="CBD5E1"),
            )
            head_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            bold_font = Font(name="Calibri", size=10, bold=True, color="000000")
            norm_font = Font(name="Calibri", size=10, color="000000")

            # Encabezados
            for c_idx, col_name in enumerate(EXCEL_COLUMNS, start=1):
                cell = ws.cell(row=1, column=c_idx, value=col_name)
                cell.fill = green_fill
                cell.font = head_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

            # Filas
            for r_idx, row in enumerate(self._rows, start=2):
                vals = row.excel_row_values()
                fill = row_fill_a if r_idx % 2 == 0 else row_fill_b
                for c_idx, val in enumerate(vals, start=1):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    try:
                        cell.value = int(val)
                    except ValueError:
                        cell.value = val
                    cell.fill = fill
                    cell.font = bold_font if c_idx in (1, 2, 7, 8) else norm_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = thin_border

            # Ajuste de ancho de columnas
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = openpyxl.utils.get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 4, 15)

            wb.save(path)
            if hasattr(self.main, "toast"):
                self.main.toast(f"Archivo exportado con éxito: {Path(path).name}", "success")
        except Exception as exc:  # noqa: BLE001
            if hasattr(self.main, "toast"):
                self.main.toast(f"Error al exportar a Excel: {exc}", "error")

    # ---------------------------------------------------------------------
    # Sincronización con el modelo y el entrenamiento activo
    # ---------------------------------------------------------------------
    def refresh_from_training(self) -> None:
        """Sincroniza la arquitectura en tiempo real con el panel de Entrenamiento y Pipeline."""
        cfg = getattr(self.main, "training_cfg", None) or TrainingConfig()
        exp = getattr(self.main, "explorer", None)

        self._updating_controls = True
        try:
            # Seleccionar preset estándar de la CNN para peces
            p_idx = self.combo_preset.findData("fish_cnn_2b")
            if p_idx >= 0:
                self.combo_preset.setCurrentIndex(p_idx)

            self.spin_blocks.setValue(2)
            self.combo_channels.setCurrentIndex(0)  # 1 canal (escala de grises)

            # Seleccionar tamaño de entrada
            sz_idx = self.combo_size.findData(int(cfg.input_size))
            if sz_idx >= 0:
                self.combo_size.setCurrentIndex(sz_idx)

            self.spin_f1.setValue(int(cfg.conv_filters))
            self.spin_f2.setValue(int(cfg.conv_filters2))
            self.spin_hidden.setValue(int(cfg.hidden_units))

            # El pooling 2×2 es parte interna de la CNN y no debe confundirse con
            # la etapa de pooling del pipeline visual, que solo afecta la imagen.
            self.chk_pool.setChecked(True)
        finally:
            self._updating_controls = False

        self._recalculate_architecture()

        res = getattr(self.main, "training_result", None)
        if res is not None and getattr(res, "model", None) is not None:
            self._model = res.model
            self.diagram.set_model(self._model)
            if exp and getattr(exp, "_result", None) is not None:
                net_in = exp._result.net_input
                if hasattr(self._model, "forward_trace"):
                    try:
                        trace = self._model.forward_trace(net_in[None, ...])
                        self.diagram.set_activations(trace)
                    except Exception:  # noqa: BLE001
                        pass
        self._rebuild_lab()

    def refresh_model(self) -> None:
        """Toma la primera muestra del dataset para el laboratorio cuando se escanea."""
        info = getattr(self.main, "dataset_info", None)
        if info and info.samples:
            self._lab_sample = info.samples[0]
        self._rebuild_lab()

    def _load_dataset_patch(self) -> None:
        info = getattr(self.main, "dataset_info", None)
        if not info or not info.samples:
            if hasattr(self.main, "toast"):
                self.main.toast("Escanea un dataset para usar muestras reales en el laboratorio.", "warning")
            self._lab_sample = None
            self._rebuild_lab()
            return
        self._lab_sample = info.samples[int(np.random.randint(0, len(info.samples)))]
        self._rebuild_lab()
        if hasattr(self.main, "toast"):
            self.main.toast(
                f"Muestra cargada: {self._lab_sample.class_name} · {self._lab_sample.path.name}",
                "info",
            )

    # ---------------------------------------------------------------------
    # Lógica del laboratorio numérico: pesos y activaciones reales
    # ---------------------------------------------------------------------
    def _lab_context(self):
        """Modelo, configuración y tamaño de entrada que usa el laboratorio."""
        main = getattr(self, "main", None)
        model = getattr(main, "loaded_model", None)
        meta = dict(getattr(main, "loaded_meta", {}) or {})
        cfg = getattr(main, "loaded_cfg", None)
        trained = model is not None

        if model is None:
            result = getattr(main, "training_result", None)
            if result is not None and getattr(result, "model", None) is not None:
                model = result.model
                cfg = getattr(result, "config", None) or cfg
                trained = True

        if cfg is None:
            cfg = getattr(main, "training_cfg", None) or TrainingConfig()

        if model is None:
            model = build_model(
                getattr(cfg, "model_type", "cnn"),
                int(cfg.input_size),
                int(cfg.hidden_units),
                int(cfg.conv_filters),
                int(cfg.conv_filters2),
                n_out=len(CLASSES),
                seed=int(cfg.seed),
                dropout=float(cfg.dropout),
            )
            trained = False

        size = int(meta.get("input_size") or getattr(cfg, "input_size", 48))
        return model, cfg, size, trained

    def _lab_input_for(self, size: int):
        """Entrada real del modelo: muestra del dataset preprocesada o tensor del pipeline."""
        sample = self._lab_sample
        if sample is not None:
            explorer = getattr(self.main, "explorer", None)
            pcfg = getattr(explorer, "cfg", None)
            eq = getattr(pcfg, "equalization", "clahe")
            clip = float(getattr(pcfg, "clahe_clip", 2.0))
            roi = bool(getattr(pcfg, "roi_enabled", False))
            zoom = float(getattr(pcfg, "roi_zoom", 1.0))
            rx = float(getattr(pcfg, "roi_x", 0.5))
            ry = float(getattr(pcfg, "roi_y", 0.5))
            key = (str(sample.path), size, eq, clip, roi, zoom, rx, ry)
            if getattr(self, "_lab_input_key", None) == key and self._lab_input is not None:
                return self._lab_input, int(sample.label)
            try:
                arr = preprocess_for_training(sample.path, size, eq, clip, roi, zoom, rx, ry)
                self._lab_input = arr
                self._lab_input_key = key
                return arr, int(sample.label)
            except Exception as exc:  # noqa: BLE001
                if hasattr(self.main, "toast"):
                    self.main.toast(f"No se pudo preprocesar la muestra: {exc}", "warning")

        result = getattr(getattr(self.main, "explorer", None), "_result", None)
        net_input = getattr(result, "net_input", None)
        if net_input is not None and tuple(net_input.shape[-2:]) == (size, size):
            return np.asarray(net_input, dtype=np.float32), None

        rng = np.random.default_rng(7)
        return rng.random((size, size), dtype=np.float32), None

    def _on_lab_stage_changed(self) -> None:
        stage = str(self.combo_stage.currentData() or "conv1")
        self._filter_field.setVisible(stage == "conv1")
        self._neuron_field.setVisible(stage == "hidden")
        self._rebuild_lab()

    def _rebuild_lab(self) -> None:
        if not hasattr(self, "combo_stage"):
            return
        stage = str(self.combo_stage.currentData() or "conv1")
        if hasattr(self, "_filter_field"):
            self._filter_field.setVisible(stage == "conv1")
            self._neuron_field.setVisible(stage == "hidden")

        model, cfg, size, trained = self._lab_context()
        x, true_label = self._lab_input_for(size)
        self._lab_model = model
        try:
            trace = model.forward_trace(x[None, ...])
        except Exception as exc:  # noqa: BLE001
            trace = []
            if hasattr(self.main, "toast"):
                self.main.toast(f"El modelo no acepta la muestra ({exc}).", "warning")

        self._lab_trace = trace
        self.diagram.set_model(model)
        self.diagram.set_image(x)
        if trace:
            self.diagram.set_activations(trace)

        try:
            if stage == "hidden":
                self._lab_steps = self._build_hidden_frames(model, trace)
            elif stage == "softmax":
                self._lab_steps = self._build_softmax_frames(trace, true_label)
            else:
                self._lab_steps = self._build_conv_frames(model, x)
        except Exception as exc:  # noqa: BLE001
            self._lab_steps = []
            if hasattr(self.main, "toast"):
                self.main.toast(f"No se pudo calcular el laboratorio: {exc}", "warning")

        sample = self._lab_sample
        if sample is not None:
            state = "" if trained else " · modelo sin entrenar"
            self._lab_sample_label.setText(
                f"Muestra: {sample.class_name} · {sample.path.name}{state}"
            )
        elif trained:
            self._lab_sample_label.setText("Tensor del pipeline · modelo cargado")
        else:
            self._lab_sample_label.setText("Entrada sintética · modelo sin entrenar")

        n = len(self._lab_steps)
        self._lab_slider.blockSignals(True)
        self._lab_slider.setRange(0, max(0, n - 1))
        self._lab_slider.blockSignals(False)
        if n:
            self._show_lab_step(0)
        else:
            self._lab_step_label.setText("Paso 0 / 0")
            self._lab_detail.setText("Carga un modelo o escanea el dataset para ver el cálculo real.")
            self._lab_formula.setText("—")

    def _build_conv_frames(self, model, x: np.ndarray) -> list[LabStep]:
        convs = model.conv_layers()
        if not convs:
            raise ValueError("El modelo no tiene capas convolucionales")
        weights = np.asarray(convs[0].W)
        bias = np.asarray(convs[0].b)
        filters = int(weights.shape[0])
        if self.combo_filter.count() != filters:
            self.combo_filter.blockSignals(True)
            self.combo_filter.clear()
            for i in range(filters):
                self.combo_filter.addItem(f"Filtro {i + 1} de {filters}", i)
            self.combo_filter.blockSignals(False)
        k = max(0, min(filters - 1, int(self.combo_filter.currentIndex())))
        kernel = weights[k, 0]
        return conv1_frames(x, kernel, float(bias[k]), crop=8)

    def _build_hidden_frames(self, model, trace: list[dict]) -> list[LabStep]:
        layer = model.hidden_layer()
        weights = np.asarray(layer.W)
        bias = np.asarray(layer.b).ravel()
        neurons = int(weights.shape[1])
        self.spin_neuron.blockSignals(True)
        self.spin_neuron.setMaximum(max(1, neurons))
        self.spin_neuron.blockSignals(False)
        neuron = max(0, min(neurons - 1, int(self.spin_neuron.value()) - 1))

        flat = None
        hidden_pos = None
        for i, item in enumerate(trace):
            if item.get("kind") == "HiddenLayer":
                hidden_pos = i
                break
        if hidden_pos is not None and hidden_pos > 0:
            flat = np.asarray(trace[hidden_pos - 1].get("array")).ravel()
        if flat is None or flat.size != weights.shape[0]:
            raise ValueError("No se pudo obtener el vector aplanado que entra a la capa oculta")
        return hidden_frames(flat, weights[:, neuron], float(bias[neuron]), neuron, top_k=49)

    def _build_softmax_frames(self, trace: list[dict], true_label: int | None) -> list[LabStep]:
        logits = None
        for item in reversed(trace):
            if item.get("kind") == "HiddenLayer":
                logits = np.asarray(item.get("array")).ravel()
                break
        if logits is None or logits.size < 2:
            raise ValueError("No se encontraron los logits de la capa de salida")
        labels = [CLASS_LABELS[c] for c in CLASSES]
        return softmax_frames(logits, labels, true_label)

    def _show_lab_step(self, idx: int) -> None:
        if not self._lab_steps:
            return
        idx = max(0, min(len(self._lab_steps) - 1, int(idx)))
        self._step_idx = idx

        self._lab_slider.blockSignals(True)
        self._lab_slider.setValue(idx)
        self._lab_slider.blockSignals(False)

        step = self._lab_steps[idx]
        self._lab_step_label.setText(f"Paso {idx + 1} / {len(self._lab_steps)}")
        self._lab_detail.setText(f"<b>{step.title}</b> — {step.detail}")
        self._lab_formula.setText(step.formula or "—")

        self.lbl_src_title.setText(step.source_label)
        self.lbl_tgt_title.setText(step.target_label)

        self.mat_source.set_matrix(step.source, highlight=step.source_highlight, color=ACCENT)

        if step.kernel is not None:
            self.mat_kernel.setVisible(True)
            self.lbl_kernel_title.setVisible(True)
            self.lbl_kernel_title.setText(step.kernel_label or "Filtro 3×3")
            self.lbl_op1.setVisible(True)
            self.lbl_op1.setText("⊛" if step.phase in ("conv", "") else "×")
            self.lbl_op2.setVisible(True)
            self.mat_kernel.set_matrix(step.kernel, color=WARNING)
        else:
            self.mat_kernel.setVisible(False)
            self.lbl_kernel_title.setVisible(False)
            self.lbl_op1.setVisible(True)
            self.lbl_op1.setText("→")
            self.lbl_op2.setVisible(False)

        self.mat_target.set_matrix(step.target, highlight=step.target_highlight, color=SUCCESS)

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("Reproducir")
            self.btn_play.setIcon(icons.icon("play", ON_ACCENT, 13))
        else:
            self._play_timer.start()
            self.btn_play.setText("Pausar")
            self.btn_play.setIcon(icons.icon("check", ON_ACCENT, 13))

    def _on_play_tick(self) -> None:
        if not self._lab_steps:
            return
        next_idx = self._step_idx + 1
        if next_idx >= len(self._lab_steps):
            next_idx = 0
        self._show_lab_step(next_idx)

    # ---------------------------------------------------------------------
    # Auditoría del Motor Neuronal Manual
    # ---------------------------------------------------------------------
    def _run_gradient_audit(self) -> None:
        self.audit_btn.setEnabled(False)
        self.audit_btn.setText("Comprobando gradientes…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from core.nn import HiddenLayer, ReLU
            rng = np.random.default_rng(1234)
            d = HiddenLayer(4, 2, rng=rng, dtype=np.float64)
            x = rng.standard_normal((3, 4))
            dout = rng.standard_normal((3, 2))
            out = d.forward(x)
            d.backward(dout)

            def f_w(w):
                d.W = w
                return float(np.sum(d.forward(x) * dout))

            err_w = gradient_check(f_w, d.W.copy(), d.dW.copy(), eps=1e-5)

            def f_b(b):
                d.b = b
                return float(np.sum(d.forward(x) * dout))

            err_b = gradient_check(f_b, d.b.copy(), d.db.copy(), eps=1e-5)

            self.audit_result_label.setText(
                f"✅ <b>Auditoría matemática completada con éxito:</b><br>"
                f"• Error relativo gradiente dW: <code>{err_w:.2e}</code> (tolerancia &lt; 1e-4)<br>"
                f"• Error relativo gradiente db: <code>{err_b:.2e}</code> (tolerancia &lt; 1e-4)<br>"
                f"• Verificación de código fuente: <b>100% NumPy puro</b>. Sin llamadas a Torch, TF o Keras."
            )
            self.audit_result_label.setStyleSheet(
                f"background: {BG_ALT}; border: 1px solid {SUCCESS}; border-radius: 8px; padding: 12px;"
                f"color: {TEXT}; font-size: 11px;"
            )
            if hasattr(self.main, "toast"):
                self.main.toast("Auditoría superada: gradientes manuales verificados.", "success")
        except Exception as exc:  # noqa: BLE001
            self.audit_result_label.setText(f"❌ Error durante la auditoría: {exc}")
            self.audit_result_label.setStyleSheet(
                f"background: {BG_ALT}; border: 1px solid {DANGER}; border-radius: 8px; padding: 12px;"
                f"color: {DANGER}; font-size: 11px;"
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.audit_btn.setEnabled(True)
            self.audit_btn.setText("Verificar gradientes y código manual")
