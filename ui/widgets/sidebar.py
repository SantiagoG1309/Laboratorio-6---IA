"""Barra lateral de navegación."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.config import APP_VERSION, CLASSES, CLASS_COLORS, CLASS_LABELS, UNIVERSITY_CAMPUS
from ui import icons
from ui.theme import ACCENT, BORDER, FONT_DISPLAY, MUTED, MUTED_2, SUCCESS, TEXT


class NavButton(QPushButton):
    def __init__(self, text: str, icon_name: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setProperty("variant", "nav")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIcon(icons.icon(icon_name, MUTED, 20))
        self.setIconSize(QSize(20, 20))
        self.setMinimumHeight(44)
        self._icon_name = icon_name

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(checked)
        color = ACCENT if checked else MUTED
        self.setIcon(icons.icon(self._icon_name, color, 20))


class Sidebar(QFrame):
    pageChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        # Ancho flexible dentro de un rango: se adapta a ventanas pequeñas sin
        # comerse el área de contenido en pantallas grandes.
        self.setMinimumWidth(208)
        self.setMaximumWidth(252)
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(6)

        # Logo
        brand = QHBoxLayout()
        brand.setSpacing(11)
        logo = QLabel()
        logo.setFixedSize(46, 46)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(icons.logo_pixmap(44, 1.0))
        logo.setStyleSheet("background: transparent;")
        logo.setToolTip("AquaCNN: pez + red neuronal, identidad del laboratorio.")
        brand.addWidget(logo)
        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        name = QLabel("AquaCNN")
        name.setStyleSheet(f"font-family: {FONT_DISPLAY}; font-size: 19px; font-weight: 700; color: {TEXT};")
        name.setToolTip("Clasificador de peces con una red neuronal convolucional implementada en NumPy.")
        sub = QLabel("Laboratorio 6 · Visión")
        sub.setProperty("subtle", True)
        brand_col.addWidget(name)
        brand_col.addWidget(sub)
        brand.addLayout(brand_col, 1)
        layout.addLayout(brand)
        layout.addSpacing(20)

        section = QLabel("ESPACIO DE TRABAJO")
        section.setProperty("subtle", True)
        section.setStyleSheet(f"font-size: 9px; letter-spacing: 1.2px; color: {MUTED_2};")
        layout.addWidget(section)
        layout.addSpacing(4)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[NavButton] = []
        items = [
            ("Inicio", "home", "Estado del dataset y división train/val/test"),
            ("Pipeline visual", "layers", "Cada etapa del tratamiento de imagen en vivo"),
            ("Arquitectura NN", "brain", "Esquema, tablas y laboratorio numérico paso a paso"),
            ("Entrenamiento", "cpu", "Hiperparámetros y monitoreo del aprendizaje"),
            ("Resultados", "chart", "Métricas, matriz de confusión e inspector de pesos"),
            ("Predicción", "sparkles", "Arrastra una imagen y obtén la clase con su confianza"),
        ]
        for i, (text, icon_name, tip) in enumerate(items):
            btn = NavButton(text, icon_name)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, idx=i: self.pageChanged.emit(idx))
            self.group.addButton(btn, i)
            self.buttons.append(btn)
            layout.addWidget(btn)

        layout.addSpacing(16)
        section2 = QLabel("SISTEMA")
        section2.setProperty("subtle", True)
        section2.setStyleSheet(f"font-size: 9px; letter-spacing: 1.2px; color: {MUTED_2};")
        layout.addWidget(section2)
        layout.addSpacing(4)
        about = NavButton("Acerca del laboratorio", "info")
        about.setToolTip("Teoría, fórmulas, autores y créditos del proyecto")
        about.clicked.connect(lambda: self.pageChanged.emit(6))
        self.group.addButton(about, 6)
        self.buttons.append(about)
        layout.addWidget(about)
        settings = NavButton("Configuración", "sliders")
        settings.setToolTip("Tema, color de acento, animaciones, tooltips y preferencias")
        settings.clicked.connect(lambda: self.pageChanged.emit(7))
        self.group.addButton(settings, 7)
        self.buttons.append(settings)
        layout.addWidget(settings)

        layout.addStretch(1)

        # Estado del dataset
        self.status_panel = QFrame()
        self.status_panel.setProperty("soft", True)
        sp = QVBoxLayout(self.status_panel)
        sp.setContentsMargins(12, 11, 12, 11)
        sp.setSpacing(7)
        top = QHBoxLayout()
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background: {SUCCESS}; border-radius: 4px;")
        top.addWidget(dot)
        self.status_label = QLabel("Dataset listo")
        self.status_label.setStyleSheet("font-size: 11px; font-weight: 600;")
        top.addWidget(self.status_label)
        top.addStretch(1)
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: 700;")
        top.addWidget(self.count_label)
        sp.addLayout(top)
        self.class_row = QHBoxLayout()
        self.class_row.setSpacing(4)
        self.class_labels: dict[str, QLabel] = {}
        for cls in CLASSES:
            chip = QLabel(f"{CLASS_LABELS[cls][:3]} 0")
            chip.setStyleSheet(
                f"color: {CLASS_COLORS[cls]}; background: rgba(255,255,255,0.03);"
                "border-radius: 7px; padding: 2px 5px; font-size: 9px;"
            )
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.class_labels[cls] = chip
            self.class_row.addWidget(chip)
        sp.addLayout(self.class_row)
        layout.addWidget(self.status_panel)

        # Universidad
        uni = QHBoxLayout()
        uni.setSpacing(8)
        uni.setContentsMargins(2, 4, 0, 2)
        shield = QLabel()
        shield.setFixedSize(30, 30)
        shield.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shield.setPixmap(icons.asset_pixmap("udec_escudo.png", 28))
        shield.setToolTip("Escudo de la Universidad de Cundinamarca.")
        uni.addWidget(shield)
        uni_col = QVBoxLayout()
        uni_col.setSpacing(0)
        uni_name = QLabel("Universidad de")
        uni_name.setProperty("subtle", True)
        uni_name.setStyleSheet("font-size: 9px; letter-spacing: 0.4px;")
        uni_col.addWidget(uni_name)
        uni_name2 = QLabel("Cundinamarca")
        uni_name2.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {TEXT};")
        uni_col.addWidget(uni_name2)
        uni_campus = QLabel(UNIVERSITY_CAMPUS)
        uni_campus.setProperty("subtle", True)
        uni_campus.setStyleSheet("font-size: 9px; letter-spacing: 0.2px;")
        uni_col.addWidget(uni_campus)
        uni.addLayout(uni_col, 1)
        layout.addLayout(uni)

        version = QLabel(f"AquaCNN {APP_VERSION} · PySide6 + NumPy")
        version.setProperty("subtle", True)
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        # El contenido completo vive dentro de un área desplazable: en
        # ventanas bajas la tarjeta del dataset ya no se recorta ni se solapa
        # con el pie de la barra lateral.
        inner = QWidget()
        inner.setLayout(layout)
        inner.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)

        self.set_current(0)

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)

    def set_dataset(self, total: int, counts: dict[str, int]) -> None:
        self.count_label.setText(str(total))
        if total > 0:
            self.status_label.setText("Dataset cargado")
        else:
            self.status_label.setText("Sin dataset")
        for cls, lbl in self.class_labels.items():
            lbl.setText(f"{CLASS_LABELS[cls][:3]} {counts.get(cls, 0)}")
