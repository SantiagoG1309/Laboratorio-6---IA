"""Página de inicio: bienvenida, estado del dataset y división train/val/test."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import (
    CLASSES,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_SCIENTIFIC,
    MAX_DATASET_IMAGES,
    UNIVERSITY_CAMPUS,
)
from core.dataset import DatasetInfo
from ui import icons
from ui.theme import (
    ACCENT,
    ACCENT_2,
    BG_ALT,
    BORDER,
    BORDER_SOFT,
    MONO_STACK,
    MUTED,
    ON_ACCENT,
    PURPLE,
    SUCCESS,
    SURFACE_2,
    TEXT,
    WARNING,
    rgba,
)
from ui.widgets.cards import Badge, Card, Divider, EmptyState, SectionHeader, StatCard, ToggleSwitch
from ui.widgets.charts import ClassBarsCanvas
from ui.widgets.dropzone import DropZone
from ui.widgets.flow_layout import ResponsiveGrid
from ui.widgets.image_view import ImageView


class ClassCard(Card):
    def __init__(self, class_name: str, parent: QWidget | None = None):
        super().__init__(parent, padding=15, spacing=8)
        self.class_name = class_name
        self.setMinimumWidth(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        color = CLASS_COLORS[class_name]
        head = QHBoxLayout()
        dot = QLabel()
        dot.setFixedSize(9, 9)
        dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        head.addWidget(dot)
        title = QLabel(CLASS_LABELS[class_name])
        title.setProperty("h3", True)
        head.addWidget(title)
        head.addStretch(1)
        self.count_badge = QLabel("0")
        self.count_badge.setStyleSheet(
            f"color: {color}; background: {rgba(color, 0.09)}; border: 1px solid {rgba(color, 0.27)};"
            "border-radius: 9px; padding: 1px 8px; font-size: 11px; font-weight: 700;"
        )
        head.addWidget(self.count_badge)
        self.body().addLayout(head)
        scientific = QLabel(CLASS_SCIENTIFIC[class_name])
        scientific.setProperty("subtle", True)
        scientific.setStyleSheet("font-style: italic;")
        self.body().addWidget(scientific)

        self.thumbs_row = QHBoxLayout()
        self.thumbs_row.setSpacing(6)
        self.thumbs: list[ImageView] = []
        for _ in range(3):
            iv = ImageView("", "image", min_height=64, rounded=8)
            iv.setFixedHeight(64)
            iv.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.thumbs_row.addWidget(iv)
            self.thumbs.append(iv)
        self.body().addLayout(self.thumbs_row)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet(
            f"QProgressBar {{ background: {SURFACE_2}; border: none; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
        )
        self.body().addWidget(self.bar)

        foot = QHBoxLayout()
        self.percent = QLabel("0 % del dataset")
        self.percent.setProperty("subtle", True)
        foot.addWidget(self.percent)
        foot.addStretch(1)
        self.add_btn = QPushButton("Añadir imágenes")
        self.add_btn.setProperty("variant", "ghost")
        self.add_btn.setIcon(icons.icon("plus", MUTED, 14))
        self.add_btn.setStyleSheet("padding: 4px 9px; font-size: 11px;")
        foot.addWidget(self.add_btn)
        self.body().addLayout(foot)

    def update_data(self, count: int, total: int, thumb_paths: list[str]) -> None:
        color = CLASS_COLORS[self.class_name]
        self.count_badge.setText(str(count))
        pct = (count / total * 100) if total else 0
        self.bar.setValue(int(pct))
        self.percent.setText(f"{count} imágenes · {pct:.0f} %")
        for i, iv in enumerate(self.thumbs):
            if i < len(thumb_paths):
                pm = QPixmap(thumb_paths[i])
                iv.set_image(pm if not pm.isNull() else None)
            else:
                iv.clear()


class PipelineStrip(Card):
    steps = [
        ("image", "Adquisición", "Imagen RGB"),
        ("layers", "Espacios de color", "R · G · B / HSV"),
        ("maximize", "Preprocesamiento", "Otsu · margen"),
        ("wave", "Análisis", "Histograma"),
        ("grid", "Convolución", "Kernels 3×3"),
        ("target", "Pooling + Flatten", "Vector de entrada"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, padding=16, spacing=12)
        head = SectionHeader("Flujo del laboratorio", "Del píxel al vector de características")
        open_btn = QPushButton("Abrir pipeline visual")
        open_btn.setProperty("variant", "primary")
        open_btn.setIcon(icons.icon("arrow", ON_ACCENT, 16))
        head.add_right(open_btn)
        self.body().addWidget(head)
        self.open_btn = open_btn
        row = QHBoxLayout()
        row.setSpacing(6)
        for i, (ic, title, sub) in enumerate(self.steps):
            box = QVBoxLayout()
            box.setSpacing(3)
            ic_lbl = QLabel()
            ic_lbl.setPixmap(icons.pixmap(ic, ACCENT if i % 2 == 0 else ACCENT_2, 22))
            ic_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(ic_lbl)
            t = QLabel(title)
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setStyleSheet("font-size: 11px; font-weight: 600;")
            box.addWidget(t)
            s = QLabel(sub)
            s.setProperty("subtle", True)
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(s)
            row.addLayout(box, 1)
            if i < len(self.steps) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet(f"color: {BORDER}; font-size: 22px; font-weight: 300;")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                row.addWidget(arrow)
        self.body().addLayout(row)


class DatasetStructureCard(Card):
    """Explica la estructura de carpetas que lee el programa y el tope de imágenes."""

    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent, padding=16, spacing=10)
        head = SectionHeader(
            "Estructura del dataset",
            "La aplicación lee las imágenes directamente dentro de estas tres carpetas",
        )
        self.limit_badge = Badge(f"Máx. {MAX_DATASET_IMAGES:,} imágenes".replace(",", "."), ACCENT)
        head.add_right(self.limit_badge)
        self.body().addWidget(head)

        row = QHBoxLayout()
        row.setSpacing(18)
        tree = QLabel(
            f'<span style="color:{ACCENT};font-weight:700;">Dataset Imagenes/</span><br>'
            f'&nbsp;&nbsp;├── <span style="color:{ACCENT_2};font-weight:600;">trucha/</span><br>'
            f'&nbsp;&nbsp;├── <span style="color:{SUCCESS};font-weight:600;">mojarra/</span><br>'
            f'&nbsp;&nbsp;└── <span style="color:{WARNING};font-weight:600;">pavon/</span>'
        )
        tree.setTextFormat(Qt.TextFormat.RichText)
        tree.setStyleSheet(
            f"font-family: {MONO_STACK}; font-size: 12.5px;"
            f"background: {BG_ALT}; border: 1px solid {BORDER_SOFT}; border-radius: 10px; padding: 12px 14px;"
        )
        tree.setMinimumWidth(250)
        row.addWidget(tree, 0)

        notes = QVBoxLayout()
        notes.setSpacing(4)
        for text, color in (
            ("Cualquier cantidad de imágenes: desde 1 hasta 10.000 en total (también menos de 100).", ACCENT),
            ("Los nombres deben ser exactamente trucha, mojarra y pavon (sin tildes).", TEXT),
            ("Formatos: .jpg · .jpeg · .png · .bmp · .webp · .tif · .tiff", MUTED),
            ("Las imágenes van sueltas dentro de cada carpeta; las subcarpetas no se leen.", MUTED),
        ):
            lbl = QLabel("•  " + text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
            notes.addWidget(lbl)
        notes.addStretch(1)
        row.addLayout(notes, 1)
        self.body().addLayout(row)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.limit_note = QLabel("")
        self.limit_note.setWordWrap(True)
        self.limit_note.setStyleSheet(f"color: {MUTED}; font-size: 10.5px;")
        foot.addWidget(self.limit_note, 1)
        self.open_btn = QPushButton("Abrir carpeta")
        self.open_btn.setProperty("variant", "ghost")
        self.open_btn.setIcon(icons.icon("folder", MUTED, 14))
        self.create_btn = QPushButton("Crear estructura")
        self.create_btn.setProperty("variant", "primary")
        self.create_btn.setIcon(icons.icon("plus", ON_ACCENT, 14))
        self.open_btn.clicked.connect(main.open_dataset_folder)
        self.create_btn.clicked.connect(main.create_structure)
        foot.addWidget(self.open_btn)
        foot.addWidget(self.create_btn)
        self.body().addLayout(foot)

    def set_info(self, info: DatasetInfo) -> None:
        if info.truncated:
            self.limit_note.setText(
                f"Se encontraron {info.found} imágenes y se usarán {info.total} "
                f"(tope de {info.limit}). Reduce las carpetas para procesar el resto."
            )
            self.limit_note.setStyleSheet(f"color: {WARNING}; font-size: 10.5px; font-weight: 600;")
        else:
            self.limit_note.setText(
                f"{info.total} de {info.limit} imágenes disponibles · "
                "el tope evita entrenamientos inviables en CPU."
            )
            self.limit_note.setStyleSheet(f"color: {MUTED}; font-size: 10.5px;")


class DashboardPage(QWidget):
    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent)
        self.main = main
        self._info: DatasetInfo | None = None
        self._thumbs: dict[str, str] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        content.setObjectName("Page")
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(26, 22, 26, 28)
        layout.setSpacing(18)

        # Hero
        hero = QFrame()
        hero.setObjectName("Hero")
        hero_l = QHBoxLayout(hero)
        hero_l.setContentsMargins(28, 26, 28, 26)
        hero_l.setSpacing(20)
        col = QVBoxLayout()
        col.setSpacing(8)
        eyebrow_row = QHBoxLayout()
        eyebrow_row.setSpacing(10)
        eyebrow = QLabel("LABORATORIO 6 · REDES NEURONALES CONVOLUCIONALES")
        eyebrow.setStyleSheet(f"color: {ACCENT}; font-size: 10px; font-weight: 700; letter-spacing: 1.4px;")
        eyebrow_row.addWidget(eyebrow)
        uni_box = QFrame()
        uni_box.setStyleSheet(
            f"background: {rgba(ACCENT, 0.06)}; border: 1px solid {rgba(ACCENT, 0.22)};"
            "border-radius: 10px;"
        )
        ul = QHBoxLayout(uni_box)
        ul.setContentsMargins(7, 3, 10, 3)
        ul.setSpacing(6)
        uni_icon = QLabel()
        uni_icon.setPixmap(icons.asset_pixmap("udec_escudo.png", 16))
        ul.addWidget(uni_icon)
        uni_text = QLabel(f"Universidad de Cundinamarca · {UNIVERSITY_CAMPUS}")
        uni_text.setStyleSheet(f"color: {ACCENT}; font-size: 9.5px; font-weight: 600;")
        ul.addWidget(uni_text)
        uni_box.setToolTip("Universidad de Cundinamarca, Sede Facatativá · Proyecto académico de Laboratorio 6.")
        eyebrow_row.addWidget(uni_box)
        eyebrow_row.addStretch(1)
        col.addLayout(eyebrow_row)
        title = QLabel("Del píxel a la especie.")
        title.setProperty("h1", True)
        title.setStyleSheet("font-size: 32px;")
        col.addWidget(title)
        desc = QLabel(
            "Tratamiento digital de imágenes de trucha, mojarra y pavón: espacios de color, segmentación Otsu, "
            "histogramas, convolución, pooling y una red neuronal entrenada con Backpropagation + Momentum."
        )
        desc.setProperty("muted", True)
        desc.setWordWrap(True)
        desc.setMaximumWidth(720)
        desc.setStyleSheet("font-size: 13px; line-height: 1.5;")
        col.addWidget(desc)
        col.addSpacing(6)
        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.explore_btn = QPushButton("Explorar pipeline")
        self.explore_btn.setProperty("variant", "primary")
        self.explore_btn.setIcon(icons.icon("layers", ON_ACCENT, 17))
        self.train_btn = QPushButton("Entrenar modelo")
        self.train_btn.setProperty("variant", "ghost")
        self.train_btn.setIcon(icons.icon("play", ACCENT, 16))
        self.arch_btn = QPushButton("Ver arquitectura")
        self.arch_btn.setProperty("variant", "ghost")
        self.arch_btn.setIcon(icons.icon("brain", ACCENT, 16))
        btns.addWidget(self.explore_btn)
        btns.addWidget(self.arch_btn)
        btns.addWidget(self.train_btn)
        btns.addStretch(1)
        col.addLayout(btns)
        hero_l.addLayout(col, 1)
        art = QLabel()
        art.setPixmap(icons.logo_pixmap(150, 1.0))
        art.setStyleSheet("background: transparent;")
        art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        art.setToolTip("AquaCNN: clasificación de peces con convolución, ReLU, pooling y una capa oculta.")
        hero_l.addWidget(art, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(hero)

        # Stats
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.stat_total = StatCard("database", "Imágenes totales", "0", ACCENT)
        self.stat_classes = StatCard("target", "Clases activas", "0 / 3", SUCCESS)
        self.stat_balance = StatCard("chart", "Balance del dataset", "—", WARNING)
        self.stat_split = StatCard("layers", "Train / Val / Test", "—", PURPLE)
        self.stat_total.setToolTip("Total de imágenes detectadas en las tres carpetas de clase.")
        self.stat_classes.setToolTip("Clases con al menos una imagen válida (trucha, mojarra y pavón).")
        self.stat_balance.setToolTip("Qué tan equilibradas están las clases: un dataset balanceado evita sesgos.")
        self.stat_split.setToolTip("División estratificada: entrenamiento / validación / prueba, respetando proporciones.")
        for w in (self.stat_total, self.stat_classes, self.stat_balance, self.stat_split):
            stats_row.addWidget(w, 1)
        layout.addLayout(stats_row)

        # Dataset / empty
        self.dataset_header = SectionHeader("Clases del dataset", "Estructura: Dataset Imagenes/<clase>/")
        self.dataset_card = Card(padding=18, spacing=12)
        self.dataset_card.body().addWidget(self.dataset_header)
        self.empty_widget = EmptyState(
            "database",
            "Tu dataset está vacío",
            "Crea la estructura de carpetas y arrastra imágenes de trucha, mojarra y pavón. "
            "La aplicación también puede copiar el dataset de ejemplo incluido.",
        )
        self.select_btn = QPushButton("Seleccionar carpeta")
        self.select_btn.setProperty("variant", "primary")
        self.select_btn.setIcon(icons.icon("folder", ON_ACCENT, 16))
        self.demo_btn = QPushButton("Usar dataset de ejemplo")
        self.demo_btn.setProperty("variant", "ghost")
        self.demo_btn.setIcon(icons.icon("sparkles", ACCENT, 15))
        self.structure_btn = QPushButton("Crear estructura")
        self.structure_btn.setProperty("variant", "ghost")
        self.structure_btn.setIcon(icons.icon("plus", MUTED, 15))
        self.empty_widget.add_button(self.select_btn)
        self.empty_widget.add_button(self.demo_btn)
        self.empty_widget.add_button(self.structure_btn)
        self.drop_zone = DropZone()
        self.empty_box = QWidget()
        eb = QVBoxLayout(self.empty_box)
        eb.setContentsMargins(0, 0, 0, 0)
        eb.setSpacing(10)
        eb.addWidget(self.empty_widget)
        eb.addWidget(self.drop_zone)
        self.dataset_card.body().addWidget(self.empty_box)
        self.class_grid = QWidget()
        self.class_grid_layout = QHBoxLayout(self.class_grid)
        self.class_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.class_grid_layout.setSpacing(12)
        self.class_cards: dict[str, ClassCard] = {}
        for cls in CLASSES:
            card = ClassCard(cls)
            card.add_btn.clicked.connect(lambda _=False, c=cls: self.main.add_images_dialog(c))
            self.class_cards[cls] = card
            self.class_grid_layout.addWidget(card)
        self.dataset_card.body().addWidget(self.class_grid)
        layout.addWidget(self.dataset_card)

        self.structure_card = DatasetStructureCard(self.main)
        layout.addWidget(self.structure_card)

        # Split + distribution
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        self.split_card = Card(padding=18, spacing=10)
        self.split_card.setMinimumWidth(420)
        sh = SectionHeader("División del dataset", "Entrenamiento · validación · prueba")
        self.split_card.body().addWidget(sh)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.spin_train = self._ratio_spin(70, ACCENT_2)
        self.spin_val = self._ratio_spin(15, SUCCESS)
        self.spin_test = self._ratio_spin(15, WARNING)
        for label, spin, color in (
            ("Entrenamiento", self.spin_train, ACCENT_2),
            ("Validación", self.spin_val, SUCCESS),
            ("Prueba", self.spin_test, WARNING),
        ):
            box = QVBoxLayout()
            lab = QLabel(label)
            lab.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")
            box.addWidget(lab)
            box.addWidget(spin)
            row.addLayout(box, 1)
        self.split_card.body().addLayout(row)
        seed_row = QHBoxLayout()
        seed_lbl = QLabel("Semilla aleatoria")
        seed_lbl.setProperty("muted", True)
        seed_row.addWidget(seed_lbl)
        seed_row.addStretch(1)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setFixedWidth(90)
        seed_row.addWidget(self.seed_spin)
        self.split_card.body().addLayout(seed_row)
        apply_row = QHBoxLayout()
        self.split_info = QLabel("Aún no hay imágenes")
        self.split_info.setProperty("muted", True)
        self.split_info.setWordWrap(True)
        apply_row.addWidget(self.split_info, 1)
        self.apply_split_btn = QPushButton("Aplicar división")
        self.apply_split_btn.setProperty("variant", "primary")
        self.apply_split_btn.setIcon(icons.icon("refresh", ON_ACCENT, 14))
        apply_row.addWidget(self.apply_split_btn)
        self.split_card.body().addLayout(apply_row)
        bottom.addWidget(self.split_card, 3)

        self.dist_card = Card(padding=14, spacing=6)
        self.dist_chart = ClassBarsCanvas()
        self.dist_chart.setMinimumHeight(190)
        self.dist_card.body().addWidget(self.dist_chart)
        bottom.addWidget(self.dist_card, 2)
        layout.addLayout(bottom)

        self.pipeline_strip = PipelineStrip()
        layout.addWidget(self.pipeline_strip)

        # Recent
        self.recent_card = Card(padding=18, spacing=10)
        self.recent_header = SectionHeader("Muestras recientes", "Haz clic en una imagen para abrir el pipeline")
        self.recent_card.body().addWidget(self.recent_header)
        self.recent_grid = ResponsiveGrid(spacing=8, min_item_width=120)
        self.recent_card.body().addWidget(self.recent_grid)
        layout.addWidget(self.recent_card)
        layout.addStretch(1)

        # Conexiones
        self.explore_btn.clicked.connect(lambda: self.main.goto_page(1))
        self.arch_btn.clicked.connect(lambda: self.main.goto_page(2))
        self.train_btn.clicked.connect(lambda: self.main.goto_page(3))
        self.pipeline_strip.open_btn.clicked.connect(lambda: self.main.goto_page(1))
        self.select_btn.clicked.connect(self.main.choose_dataset_folder)
        self.demo_btn.clicked.connect(self.main.use_bundled_dataset)
        self.structure_btn.clicked.connect(self.main.create_structure)
        self.drop_zone.clicked.connect(self.main.choose_dataset_folder)
        self.drop_zone.pathsDropped.connect(self.main.import_dropped_images)
        self.apply_split_btn.clicked.connect(self._apply_split)

    def _ratio_spin(self, value: float, color: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setDecimals(0)
        spin.setSingleStep(5)
        spin.setValue(value)
        spin.setSuffix(" %")
        spin.setStyleSheet(f"QDoubleSpinBox {{ border-color: {rgba(color, 0.33)}; }}")
        return spin

    def _apply_split(self) -> None:
        self.main.apply_split(
            self.spin_train.value() / 100.0,
            self.spin_val.value() / 100.0,
            self.spin_test.value() / 100.0,
            self.seed_spin.value(),
        )

    def refresh(self, info: DatasetInfo, thumbs: dict[str, str], splits: dict | None = None) -> None:
        self._info = info
        self._thumbs = thumbs or {}
        total = info.total
        self.stat_total.set_value(str(total))
        self.stat_classes.set_value(f"{info.class_count} / 3")
        self.stat_balance.set_value(f"{info.balance * 100:.0f}%" if total else "—")
        if splits:
            self.stat_split.set_value(f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
        else:
            self.stat_split.set_value("—")
        self.dist_chart.update_counts(info.counts)
        self.structure_card.set_info(info)

        has_data = total > 0
        self.empty_box.setVisible(not has_data)
        self.class_grid.setVisible(has_data)
        if has_data:
            max_count = max(info.counts.values()) if info.counts else 1
            for cls in CLASSES:
                paths = [str(s.path) for s in info.samples if s.class_name == cls]
                tp = [self._thumbs[p] for p in paths if p in self._thumbs][:3]
                self.class_cards[cls].update_data(info.counts.get(cls, 0), max_count, tp)

        # Recientes
        while self.recent_grid._flow.count():  # type: ignore[attr-defined]
            item = self.recent_grid._flow.takeAt(0)  # type: ignore[attr-defined]
            if item and item.widget():
                item.widget().deleteLater()
        for sample in info.samples[:8]:
            card = QWidget()
            card.setMinimumSize(132, 108)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(3)
            iv = ImageView("", "image", min_height=86, rounded=9)
            iv.setMinimumHeight(86)
            tp = self._thumbs.get(str(sample.path))
            if tp:
                pm = QPixmap(tp)
                iv.set_image(pm if not pm.isNull() else None)
            iv.doubleClicked.connect(lambda p=str(sample.path): self.main.open_explorer(p))
            lay.addWidget(iv)
            name = QLabel(f"{CLASS_LABELS[sample.class_name]} · {sample.name[:18]}")
            name.setProperty("subtle", True)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(name)
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.mousePressEvent = lambda ev, p=str(sample.path): self.main.open_explorer(p)  # type: ignore[assignment]
            self.recent_grid.add(card)

        if splits:
            self.split_info.setText(
                f"<span style='color:{ACCENT_2}'>Train {len(splits['train'])}</span> · "
                f"<span style='color:{SUCCESS}'>Val {len(splits['val'])}</span> · "
                f"<span style='color:{WARNING}'>Test {len(splits['test'])}</span>"
            )
        else:
            self.split_info.setText("Aplica la división para preparar el entrenamiento." if has_data else "Aún no hay imágenes")
