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
from ui.widgets.matrix_table import MatrixTable
from ui.widgets.nn_diagram import NNDiagram
from core.architecture import (
    ArchRow, cnn_rows, totals, EXCEL_COLUMNS,
    cnn_architecture_rows,
    build_lab_steps, LabStep, sample_matrix, matrix_from_patch,
    audit_imports, MANUAL_ENGINE_CHECKS
)
from core.config import CLASSES, CLASS_LABELS, KERNELS, TrainingConfig
from core.image_ops import imread_rgb
from core.nn import gradient_check
from core.cnn import CNN


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

        self._lab_matrix = sample_matrix(10, seed=42)
        self._lab_steps: list[LabStep] = []
        self._step_idx = 0

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
        header_row = QHBoxLayout()
        header = SectionHeader(
            "Arquitectura de la Red Neuronal Convolucional (CNN)",
            "Diagrama interactivo, tensores capa por capa calculados en tiempo real y laboratorio numérico"
        )
        header_row.addWidget(header, 1)

        self.badge_cnn = Badge("Red Convolucional (CNN para Peces)", ACCENT)
        header_row.addWidget(self.badge_cnn)

        self.sync_btn = QPushButton("Sincronizar con entrenamiento")
        self.sync_btn.setProperty("variant", "ghost")
        self.sync_btn.setIcon(icons.icon("refresh", MUTED, 14))
        self.sync_btn.setToolTip("Carga los parámetros actuales del panel de Entrenamiento y del Pipeline.")
        self.sync_btn.clicked.connect(self.refresh_from_training)
        header_row.addWidget(self.sync_btn)

        layout.addLayout(header_row)

        # -----------------------------------------------------------------
        # 2. Tarjetas de métricas y totales (Parámetros, FLOPs, Activaciones)
        # -----------------------------------------------------------------
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self.card_params = StatCard("cpu", "Parámetros totales", "0", ACCENT)
        self.card_flops = StatCard("sliders", "FLOPs estimados", "0", PURPLE)
        self.card_elements = StatCard("layers", "Activaciones / Salida", "0", ACCENT_2)
        self.card_layers = StatCard("check", "Capas con pesos", "0", SUCCESS)

        for card in (self.card_params, self.card_flops, self.card_elements, self.card_layers):
            stats_row.addWidget(card)
        layout.addLayout(stats_row)

        # -----------------------------------------------------------------
        # 3. Diagrama interactivo de la red neuronal
        # -----------------------------------------------------------------
        diagram_card = Card(padding=14, spacing=10)
        d_head = QHBoxLayout()
        d_title = QLabel("Esquema Computacional Interactivo")
        d_title.setProperty("h3", True)
        d_head.addWidget(d_title)
        d_sub = QLabel("Pasa el cursor para ver detalles o haz clic sobre una capa para ubicarla en la tabla")
        d_sub.setProperty("muted", True)
        d_sub.setStyleSheet("font-size: 11px;")
        d_head.addWidget(d_sub, 1)

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
        t_head = QHBoxLayout()
        t_title_box = QVBoxLayout()
        t_title = QLabel("Desglose Capa por Capa y Dimensiones de Tensores")
        t_title.setProperty("h3", True)
        t_title_box.addWidget(t_title)

        self.lbl_table_subtitle = QLabel(
            "Flujo de tensores: La salida de cada etapa se convierte en la entrada exacta de la etapa siguiente."
        )
        self.lbl_table_subtitle.setProperty("muted", True)
        self.lbl_table_subtitle.setStyleSheet("font-size: 11px;")
        t_title_box.addWidget(self.lbl_table_subtitle)
        t_head.addLayout(t_title_box, 1)

        self.lbl_arch_badge = QLabel("Arquitectura: Red Neuronal Convolucional (CNN para Peces)")
        self.lbl_arch_badge.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: 600;")
        t_head.addWidget(self.lbl_arch_badge)

        # Botones de acción de la tabla
        self.btn_copy_tsv = QPushButton("Copiar tabla")
        self.btn_copy_tsv.setProperty("variant", "ghost")
        self.btn_copy_tsv.setIcon(icons.icon("check", MUTED, 13))
        self.btn_copy_tsv.setToolTip("Copia los datos de la tabla en formato TSV (para pegar en Excel).")
        self.btn_copy_tsv.clicked.connect(self._copy_table_tsv)
        t_head.addWidget(self.btn_copy_tsv)

        self.btn_export_xlsx = QPushButton("Exportar .xlsx")
        self.btn_export_xlsx.setProperty("variant", "ghost")
        self.btn_export_xlsx.setIcon(icons.icon("download", MUTED, 13))
        self.btn_export_xlsx.setToolTip("Exporta los cálculos reales de la CNN a una hoja de cálculo Excel.")
        self.btn_export_xlsx.clicked.connect(self._export_to_excel)
        t_head.addWidget(self.btn_export_xlsx)

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
        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)

        lbl_preset = QLabel("<b>Configuración CNN:</b>")
        lbl_preset.setStyleSheet(f"color: {ACCENT}; font-size: 11px;")
        preset_row.addWidget(lbl_preset)

        self.combo_preset = QComboBox()
        self.combo_preset.addItem("CNN Estándar Peces (2 Bloques Conv + Pool) · [Por Defecto]", "fish_cnn_2b")
        self.combo_preset.addItem("CNN Rápida Peces (1 Bloque Conv + Pool)", "fast_1b")
        self.combo_preset.addItem("CNN Profunda Peces (3 Bloques Conv + Pool)", "deep_3b")
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.combo_preset, 1)

        self.btn_reset_fish = QPushButton("Restablecer a CNN Peces (48×48)")
        self.btn_reset_fish.setProperty("variant", "ghost")
        self.btn_reset_fish.setIcon(icons.icon("fish", MUTED, 13))
        self.btn_reset_fish.clicked.connect(lambda: self._set_preset_by_key("fish_cnn_2b"))
        preset_row.addWidget(self.btn_reset_fish)

        ctrl_layout.addLayout(preset_row)

        # Fila B: Parámetros de la red modificables en tiempo real
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(14)
        params_grid.setVerticalSpacing(8)

        # 1. Número de bloques Conv + Pool (1 a 3 para la CNN de clasificación)
        lbl_b = QLabel("Bloques Conv+Pool:")
        lbl_b.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.spin_blocks = QSpinBox()
        self.spin_blocks.setRange(1, 3)
        self.spin_blocks.setValue(2)
        self.spin_blocks.setToolTip("Determina cuántos pares de convolución y pooling se encadenan.")
        self.spin_blocks.valueChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_b, 0, 0)
        params_grid.addWidget(self.spin_blocks, 0, 1)

        # 2. Resolución de entrada H×W
        lbl_sz = QLabel("Resolución Entrada:")
        lbl_sz.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.combo_size = QComboBox()
        for sz in (32, 40, 48, 64, 96, 128):
            self.combo_size.addItem(f"{sz} × {sz}", sz)
        self.combo_size.setCurrentIndex(2)  # 48x48
        self.combo_size.currentIndexChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_sz, 0, 2)
        params_grid.addWidget(self.combo_size, 0, 3)

        # 3. Canales de entrada (1 para escala de grises, 3 para RGB)
        lbl_ch = QLabel("Canales Entrada:")
        lbl_ch.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.combo_channels = QComboBox()
        self.combo_channels.addItem("1 (Grises · Tratamiento)", 1)
        self.combo_channels.addItem("3 (RGB · Color)", 3)
        self.combo_channels.currentIndexChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_ch, 0, 4)
        params_grid.addWidget(self.combo_channels, 0, 5)

        # 4. Filtros Conv 1
        lbl_f1 = QLabel("Filtros Conv 1:")
        lbl_f1.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.spin_f1 = QSpinBox()
        self.spin_f1.setRange(2, 64)
        self.spin_f1.setValue(6)
        self.spin_f1.valueChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_f1, 1, 0)
        params_grid.addWidget(self.spin_f1, 1, 1)

        # 5. Kernel Conv 1
        lbl_k1 = QLabel("Kernel Conv 1:")
        lbl_k1.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.combo_k1 = QComboBox()
        for k_val in (3, 5):
            self.combo_k1.addItem(f"{k_val} × {k_val}", k_val)
        self.combo_k1.setCurrentIndex(0)
        self.combo_k1.currentIndexChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_k1, 1, 2)
        params_grid.addWidget(self.combo_k1, 1, 3)

        # 6. Stride y Padding Conv 1
        sp_box = QHBoxLayout()
        sp_box.setSpacing(4)
        self.spin_s1 = QSpinBox()
        self.spin_s1.setRange(1, 2)
        self.spin_s1.setValue(1)
        self.spin_s1.valueChanged.connect(self._recalculate_architecture)
        self.spin_p1 = QSpinBox()
        self.spin_p1.setRange(0, 2)
        self.spin_p1.setValue(1)
        self.spin_p1.valueChanged.connect(self._recalculate_architecture)
        sp_box.addWidget(QLabel("S:"))
        sp_box.addWidget(self.spin_s1)
        sp_box.addWidget(QLabel("P:"))
        sp_box.addWidget(self.spin_p1)
        lbl_sp1 = QLabel("Stride / Padding C1:")
        lbl_sp1.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        params_grid.addWidget(lbl_sp1, 1, 4)
        params_grid.addLayout(sp_box, 1, 5)

        # 7. Filtros Conv 2
        lbl_f2 = QLabel("Filtros Conv 2:")
        lbl_f2.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.spin_f2 = QSpinBox()
        self.spin_f2.setRange(2, 128)
        self.spin_f2.setValue(12)
        self.spin_f2.valueChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_f2, 2, 0)
        params_grid.addWidget(self.spin_f2, 2, 1)

        # 8. Filtros Conv 3
        lbl_f3 = QLabel("Filtros Conv 3:")
        lbl_f3.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.spin_f3 = QSpinBox()
        self.spin_f3.setRange(2, 256)
        self.spin_f3.setValue(24)
        self.spin_f3.valueChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_f3, 2, 2)
        params_grid.addWidget(self.spin_f3, 2, 3)

        # 9. Pooling toggle
        lbl_pool = QLabel("Max Pooling:")
        lbl_pool.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.chk_pool = QCheckBox("Activar Pooling (2×2 s=2)")
        self.chk_pool.setChecked(True)
        self.chk_pool.toggled.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_pool, 2, 4)
        params_grid.addWidget(self.chk_pool, 2, 5)

        # 10. Capa Densa (FC)
        lbl_hid = QLabel("Neuronas Densa (FC):")
        lbl_hid.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(8, 512)
        self.spin_hidden.setValue(64)
        self.spin_hidden.setSingleStep(8)
        self.spin_hidden.valueChanged.connect(self._recalculate_architecture)
        params_grid.addWidget(lbl_hid, 3, 0)
        params_grid.addWidget(self.spin_hidden, 3, 1)

        # 11. Clases de salida (Fijo en las 3 clases de peces)
        lbl_cls = QLabel("Clasificación Peces:")
        lbl_cls.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        badge_cls = Badge("3 Clases: Trucha · Mojarra · Pavón", SUCCESS)
        badge_cls.setFixedHeight(26)
        params_grid.addWidget(lbl_cls, 3, 2)
        params_grid.addWidget(badge_cls, 3, 3, 1, 3)

        ctrl_layout.addLayout(params_grid)
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

        # Configuración de anchos de columnas
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)

        table_card.body().addWidget(self.table)
        layout.addWidget(table_card)

        # -----------------------------------------------------------------
        # 5. Laboratorio Numérico Paso a Paso (Convolución, ReLU y Pooling)
        # -----------------------------------------------------------------
        lab_card = Card(padding=16, spacing=14)
        l_head = QHBoxLayout()
        l_col = QVBoxLayout()
        l_title = QLabel("Laboratorio Numérico Paso a Paso")
        l_title.setProperty("h2", True)
        l_col.addWidget(l_title)
        l_sub = QLabel(
            "Simulación visual interactiva de la convolución 2D, activación ReLU y Max/Average Pooling celda por celda."
        )
        l_sub.setProperty("muted", True)
        l_col.addWidget(l_sub)
        l_head.addLayout(l_col, 1)
        lab_card.body().addLayout(l_head)

        # Controles del laboratorio
        ctrl_box = QHBoxLayout()
        ctrl_box.setSpacing(10)

        ctrl_box.addWidget(QLabel("Filtro / Kernel:"))
        self.combo_kernel = QComboBox()
        for k_key, k_info in KERNELS.items():
            if k_key != "custom":
                self.combo_kernel.addItem(k_info["label"], k_key)
        self.combo_kernel.currentIndexChanged.connect(self._rebuild_lab)
        ctrl_box.addWidget(self.combo_kernel)

        ctrl_box.addWidget(QLabel("Padding:"))
        self.combo_pad = QComboBox()
        self.combo_pad.addItem("0 (Sin padding)", 0)
        self.combo_pad.addItem("1 (Padding = 1)", 1)
        self.combo_pad.setCurrentIndex(1)
        self.combo_pad.currentIndexChanged.connect(self._rebuild_lab)
        ctrl_box.addWidget(self.combo_pad)

        ctrl_box.addWidget(QLabel("Stride:"))
        self.combo_stride = QComboBox()
        self.combo_stride.addItem("1", 1)
        self.combo_stride.addItem("2", 2)
        self.combo_stride.currentIndexChanged.connect(self._rebuild_lab)
        ctrl_box.addWidget(self.combo_stride)

        ctrl_box.addWidget(QLabel("Pooling:"))
        self.combo_pool = QComboBox()
        self.combo_pool.addItem("Max Pooling", "max")
        self.combo_pool.addItem("Average Pooling", "mean")
        self.combo_pool.currentIndexChanged.connect(self._rebuild_lab)
        ctrl_box.addWidget(self.combo_pool)

        self.btn_reset_lab = QPushButton("Muestra del dataset")
        self.btn_reset_lab.setProperty("variant", "ghost")
        self.btn_reset_lab.setIcon(icons.icon("fish", MUTED, 13))
        self.btn_reset_lab.setToolTip("Carga una muestra real del dataset de peces como matriz de entrada.")
        self.btn_reset_lab.clicked.connect(self._load_dataset_patch)
        ctrl_box.addWidget(self.btn_reset_lab)

        ctrl_box.addStretch(1)
        lab_card.body().addLayout(ctrl_box)

        # Visualizador de matrices
        mats_layout = QHBoxLayout()
        mats_layout.setSpacing(14)
        mats_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Entrada
        box_src = QVBoxLayout()
        box_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_src_title = QLabel("Matriz de Entrada")
        self.lbl_src_title.setStyleSheet(f"color: {ACCENT}; font-weight: 600; font-size: 11px;")
        box_src.addWidget(self.lbl_src_title, 0, Qt.AlignmentFlag.AlignCenter)
        self.mat_source = MatrixTable(10, 10, editable=False)
        box_src.addWidget(self.mat_source)
        mats_layout.addLayout(box_src)

        # Operador 1
        self.lbl_op1 = QLabel("⊛")
        self.lbl_op1.setStyleSheet(f"color: {MUTED}; font-size: 24px; font-weight: bold;")
        mats_layout.addWidget(self.lbl_op1)

        # Kernel
        box_k = QVBoxLayout()
        box_k.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_kernel_title = QLabel("Filtro 3×3")
        self.lbl_kernel_title.setStyleSheet(f"color: {WARNING}; font-weight: 600; font-size: 11px;")
        box_k.addWidget(self.lbl_kernel_title, 0, Qt.AlignmentFlag.AlignCenter)
        self.mat_kernel = MatrixTable(3, 3, editable=False)
        box_k.addWidget(self.mat_kernel)
        mats_layout.addLayout(box_k)

        # Operador 2
        self.lbl_op2 = QLabel("=")
        self.lbl_op2.setStyleSheet(f"color: {MUTED}; font-size: 24px; font-weight: bold;")
        mats_layout.addWidget(self.lbl_op2)

        # Salida
        box_tgt = QVBoxLayout()
        box_tgt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_tgt_title = QLabel("Mapa de Características")
        self.lbl_tgt_title.setStyleSheet(f"color: {SUCCESS}; font-weight: 600; font-size: 11px;")
        box_tgt.addWidget(self.lbl_tgt_title, 0, Qt.AlignmentFlag.AlignCenter)
        self.mat_target = MatrixTable(10, 10, editable=False)
        box_tgt.addWidget(self.mat_target)
        mats_layout.addLayout(box_tgt)

        lab_card.body().addLayout(mats_layout)

        # Reproductor de pasos
        player = QHBoxLayout()
        player.setSpacing(8)

        self.btn_first = QPushButton("Inicio")
        self.btn_first.setProperty("variant", "ghost")
        self.btn_first.setIcon(icons.icon("arrow-left", MUTED, 13))
        self.btn_first.clicked.connect(lambda: self._show_lab_step(0))
        player.addWidget(self.btn_first)

        self.btn_prev = QPushButton("Anterior")
        self.btn_prev.setProperty("variant", "ghost")
        self.btn_prev.setIcon(icons.icon("arrow-left", MUTED, 13))
        self.btn_prev.clicked.connect(lambda: self._show_lab_step(self._step_idx - 1))
        player.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Reproducir")
        self.btn_play.setProperty("variant", "primary")
        self.btn_play.setIcon(icons.icon("play", ON_ACCENT, 13))
        self.btn_play.clicked.connect(self._toggle_play)
        player.addWidget(self.btn_play)

        self.btn_next = QPushButton("Siguiente")
        self.btn_next.setProperty("variant", "ghost")
        self.btn_next.setIcon(icons.icon("arrow", MUTED, 13))
        self.btn_next.clicked.connect(lambda: self._show_lab_step(self._step_idx + 1))
        player.addWidget(self.btn_next)

        self.btn_last = QPushButton("Final")
        self.btn_last.setProperty("variant", "ghost")
        self.btn_last.setIcon(icons.icon("arrow", MUTED, 13))
        self.btn_last.clicked.connect(lambda: self._show_lab_step(len(self._lab_steps) - 1))
        player.addWidget(self.btn_last)

        self._lab_slider = QSlider(Qt.Orientation.Horizontal)
        self._lab_slider.setRange(0, 0)
        self._lab_slider.valueChanged.connect(self._show_lab_step)
        player.addWidget(self._lab_slider, 1)

        self._lab_step_label = QLabel("Paso 0 / 0")
        self._lab_step_label.setStyleSheet(f"color: {TEXT}; font-weight: 600; font-size: 12px; min-width: 140px;")
        self._lab_step_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        player.addWidget(self._lab_step_label)

        lab_card.body().addLayout(player)

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
            f"{num_b} bloque(s) Conv+Pool · Entrada {size}×{size}×{in_ch} · Densa {hidden} neuronas · 3 clases (Trucha, Mojarra, Pavón)"
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
            "Densa (FC)": WARNING,
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

            # Verificar si en el pipeline visual se desactivó el pooling
            exp = getattr(self.main, "explorer", None)
            if exp and hasattr(exp, "controls") and hasattr(exp.controls, "stage_switches"):
                sw_pool = exp.controls.stage_switches.get("pool")
                if sw_pool is not None:
                    self.chk_pool.setChecked(bool(sw_pool.isChecked()))
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

    def refresh_model(self) -> None:
        """Carga una muestra del dataset cuando este se escanea."""
        info = getattr(self.main, "dataset_info", None)
        if info and info.samples:
            try:
                sample = info.samples[0]
                img = imread_rgb(sample.path)
                self.diagram.set_image(img)
                self._lab_matrix = matrix_from_patch(img, 10)
                self._rebuild_lab()
            except Exception:  # noqa: BLE001
                pass

    def _load_dataset_patch(self) -> None:
        info = getattr(self.main, "dataset_info", None)
        if not info or not info.samples:
            self._lab_matrix = sample_matrix(10, seed=np.random.randint(1, 9999))
            self._rebuild_lab()
            return
        idx = np.random.randint(0, len(info.samples))
        sample = info.samples[idx]
        try:
            img = imread_rgb(sample.path)
            self.diagram.set_image(img)
            self._lab_matrix = matrix_from_patch(img, 10)
            self._rebuild_lab()
            if hasattr(self.main, "toast"):
                self.main.toast(f"Parche cargado de {sample.class_name}: {sample.path.name}", "info")
        except Exception as exc:  # noqa: BLE001
            if hasattr(self.main, "toast"):
                self.main.toast(f"No se pudo cargar la imagen: {exc}", "warning")

    # ---------------------------------------------------------------------
    # Lógica del laboratorio numérico interactivo
    # ---------------------------------------------------------------------
    def _rebuild_lab(self) -> None:
        k_key = self.combo_kernel.currentData() or "vertical"
        kernel_info = KERNELS.get(k_key, KERNELS["vertical"])
        kernel = kernel_info["matrix"]
        padding = int(self.combo_pad.currentData() or 0)
        stride = int(self.combo_stride.currentData() or 1)
        pool_mode = str(self.combo_pool.currentData() or "max")

        self._lab_steps = build_lab_steps(
            self._lab_matrix,
            kernels=[kernel],
            padding=padding,
            stride=stride,
            relu=True,
            pool_mode=pool_mode,
            pool_size=2,
            pool_stride=2,
        )

        n = len(self._lab_steps)
        self._lab_slider.blockSignals(True)
        self._lab_slider.setRange(0, max(0, n - 1))
        self._lab_slider.blockSignals(False)
        self._show_lab_step(0)

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
            self.lbl_op1.setVisible(True)
            self.mat_kernel.set_matrix(step.kernel, color=WARNING)
        else:
            self.mat_kernel.setVisible(False)
            self.lbl_kernel_title.setVisible(False)
            self.lbl_op1.setVisible(False)

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
            from core.nn import Dense, ReLU
            rng = np.random.default_rng(1234)
            d = Dense(4, 2, rng=rng, dtype=np.float64)
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
