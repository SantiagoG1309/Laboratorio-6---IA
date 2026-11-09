"""Página «Acerca del laboratorio»: teoría, fórmulas y créditos."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from core.config import APP_AUTHORS, APP_VERSION, UNIVERSITY, UNIVERSITY_CAMPUS, UNIVERSITY_CITY
from ui import icons
from ui.theme import (
    ACCENT,
    ACCENT_2,
    BG_ALT,
    BORDER,
    MONO_STACK,
    MUTED,
    PURPLE,
    SUCCESS,
    SURFACE_2,
    TEXT,
    WARNING,
    rgba,
)
from ui.widgets.cards import Card, Divider, SectionHeader


class AboutPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
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
        layout.setSpacing(16)

        head = QHBoxLayout()
        col = QVBoxLayout()
        title = QLabel("Acerca del Laboratorio 6")
        title.setProperty("h1", True)
        col.addWidget(title)
        sub = QLabel(
            "Tratamiento de imágenes para la clasificación de peces: trucha, mojarra y pavón."
        )
        sub.setProperty("muted", True)
        col.addWidget(sub)
        head.addLayout(col, 1)
        logo = QLabel()
        logo.setPixmap(icons.logo_pixmap(64, 1.0))
        logo.setToolTip("AquaCNN: pez + red neuronal, identidad del laboratorio.")
        head.addWidget(logo)
        layout.addLayout(head)

        # Universidad y autores
        team = Card(padding=18, spacing=10)
        tr = QHBoxLayout()
        tr.setSpacing(20)
        shield = QLabel()
        shield.setPixmap(icons.asset_pixmap("udec_escudo.png", 78))
        shield.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        tr.addWidget(shield, 0, Qt.AlignmentFlag.AlignTop)

        uni_col = QVBoxLayout()
        uni_col.setSpacing(3)
        uni_title = QLabel(UNIVERSITY)
        uni_title.setProperty("h2", True)
        uni_col.addWidget(uni_title)
        uni_city = QLabel(f"{UNIVERSITY_CAMPUS} · {UNIVERSITY_CITY}")
        uni_city.setProperty("muted", True)
        uni_col.addWidget(uni_city)
        uni_fac = QLabel("Facultad de Ingeniería · Laboratorio 6 de Redes Neuronales Convolucionales")
        uni_fac.setProperty("subtle", True)
        uni_fac.setWordWrap(True)
        uni_col.addWidget(uni_fac)
        uni_col.addSpacing(6)
        uni_note = QLabel(
            "Proyecto académico desarrollado para el tratamiento digital de imágenes y la "
            "clasificación de peces (trucha, mojarra y pavón)."
        )
        uni_note.setProperty("muted", True)
        uni_note.setWordWrap(True)
        uni_note.setStyleSheet("font-size: 11px;")
        uni_col.addWidget(uni_note)
        uni_col.addStretch(1)
        tr.addLayout(uni_col, 3)

        authors_col = QVBoxLayout()
        authors_col.setSpacing(6)
        a_head = QLabel("AUTORES")
        a_head.setStyleSheet(f"color: {ACCENT}; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;")
        authors_col.addWidget(a_head)
        for i, name in enumerate(APP_AUTHORS, start=1):
            row = QHBoxLayout()
            row.setSpacing(8)
            num = QLabel(str(i))
            num.setFixedSize(22, 22)
            num.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num.setStyleSheet(
                f"background: {rgba(ACCENT, 0.08)}; color: {ACCENT}; border: 1px solid {rgba(ACCENT, 0.28)};"
                "border-radius: 7px; font-size: 10px; font-weight: 700;"
            )
            row.addWidget(num)
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {TEXT}; font-size: 12px; font-weight: 600;")
            row.addWidget(lbl, 1)
            authors_col.addLayout(row)
        authors_col.addStretch(1)
        tr.addLayout(authors_col, 2)
        team.body().addLayout(tr)
        layout.addWidget(team)

        intro = Card(padding=18, spacing=8)
        intro.body().addWidget(SectionHeader("Objetivo", "Del procesamiento de la imagen a la CNN"))
        p = QLabel(
            "Aplicar técnicas de tratamiento digital de imágenes para preparar un conjunto de datos "
            "que permita clasificar peces de acuicultura. El flujo sigue el diagrama del laboratorio: "
            "adquisición → espacios de color → preprocesamiento (ROI, redimensionado, normalización) → "
            "análisis (histogramas, ecualización/CLAHE) → filtrado y convolución → pooling → flatten → "
            "clasificador neuronal con Backpropagation, Momentum y Softmax."
        )
        p.setWordWrap(True)
        p.setProperty("muted", True)
        intro.body().addWidget(p)
        layout.addWidget(intro)

        # Etapas
        stages = Card(padding=18, spacing=10)
        stages.body().addWidget(SectionHeader("Etapas implementadas", "Cada bloque es configurable en caliente"))
        items = [
            ("1", "Adquisición", "Carga RGB (H×W×3), lectura segura de rutas unicode y caché de imágenes.", ACCENT),
            ("2", "Espacios de color", "Canales R, G, B y conversión a escala de grises y HSV (H, S, V).", ACCENT_2),
            ("3", "Preprocesamiento", "ROI cuadrada configurable, redimensionado uniforme y normalización [0,1].", SUCCESS),
            ("4", "Análisis", "Histograma de 256 bins, ecualización global y CLAHE adaptativa.", WARNING),
            ("5", "Convolución", "Kernels 3×3 (vertical, horizontal, Sobel, Laplaciano, enfoque, relieve, personalizado) con padding y stride.", PURPLE),
            ("6", "Pooling + Flatten", "Max/Average pooling configurable y vectorización del mapa de características.", ACCENT),
            ("7", "Clasificador CNN", "Red neuronal convolucional con backpropagation manual en NumPy puro: convolución 2D, max pooling, ReLU, capas ocultas, Momentum, softmax y entropía cruzada.", ACCENT_2),
        ]
        for num, title, desc, color in items:
            row = QHBoxLayout()
            row.setSpacing(10)
            badge = QLabel(num)
            badge.setFixedSize(26, 26)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                f"background: {rgba(color, 0.09)}; color: {color}; border: 1px solid {rgba(color, 0.33)};"
                "border-radius: 8px; font-weight: 700; font-size: 11px;"
            )
            row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
            txt = QVBoxLayout()
            txt.setSpacing(1)
            t = QLabel(title)
            t.setStyleSheet("font-weight: 700; font-size: 12px;")
            txt.addWidget(t)
            d = QLabel(desc)
            d.setProperty("muted", True)
            d.setWordWrap(True)
            d.setStyleSheet("font-size: 11px;")
            txt.addWidget(d)
            row.addLayout(txt, 1)
            stages.body().addLayout(row)
        layout.addWidget(stages)

        # Fórmulas
        formulas = Card(padding=18, spacing=8)
        formulas.body().addWidget(SectionHeader("Formulación", "Las ecuaciones detrás de la aplicación"))
        eqs = [
            ("Convolución", "y[i,j] = Σ_m Σ_n x[i+m, j+n] · k[m,n] + b"),
            ("ReLU", "f(x) = max(0, x)"),
            ("Max / Average Pooling", "y = max_{(m,n)∈ventana} x[m,n]   ·   y = (1/k²) Σ x[m,n]"),
            ("Softmax", "p_i = exp(z_i) / Σ_j exp(z_j)"),
            ("Entropía cruzada", "L = −(1/N) Σ_i log(p_i[y_i])"),
            ("Momentum", "v_t = μ·v_{t−1} − η·∇L ;   θ_t = θ_{t−1} + v_t"),
        ]
        for name, eq in eqs:
            row = QHBoxLayout()
            n = QLabel(name)
            n.setFixedWidth(150)
            n.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            row.addWidget(n)
            e = QLabel(eq)
            e.setStyleSheet(
                f"font-family: {MONO_STACK}; font-size: 11px; color: {TEXT};"
                f"background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 7px; padding: 5px 9px;"
            )
            row.addWidget(e, 1)
            formulas.body().addLayout(row)
        layout.addWidget(formulas)

        # Stack + créditos
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        stack = Card(padding=18, spacing=8)
        stack.body().addWidget(SectionHeader("Stack tecnológico", "Arquitectura MVC + QThread"))
        for name, desc in (
            ("PySide6", "Interfaz Qt6, tema oscuro, hilos QThread/QThreadPool."),
            ("OpenCV", "Lectura, color, ROI, redimensionado, histogramas, CLAHE."),
            ("NumPy", "Tensores, convolución vectorizada, backpropagation."),
            ("Matplotlib", "Histogramas, curvas de entrenamiento y matriz de confusión."),
            ("PyInstaller", "Empaquetado en ejecutable nativo independiente."),
        ):
            row = QHBoxLayout()
            k = QLabel(name)
            k.setFixedWidth(90)
            k.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: 700;")
            row.addWidget(k)
            v = QLabel(desc)
            v.setProperty("muted", True)
            v.setWordWrap(True)
            v.setStyleSheet("font-size: 11px;")
            row.addWidget(v, 1)
            stack.body().addLayout(row)
        bottom.addWidget(stack, 1)

        credits = Card(padding=18, spacing=8)
        credits.body().addWidget(SectionHeader("Créditos y referencias", "Uso académico"))
        c = QLabel(
            "Escudo de la Universidad de Cundinamarca: imagen institucional tomada de Wikimedia "
            "Commons (Escudo_Universidad_de_Cundinamarca.png), usada con fines académicos.\n\n"
            "Dataset de ejemplo: fotografías de Wikimedia Commons e iNaturalist con licencias "
            "Creative Commons (ver CREDITOS.txt y CREDITOS_INATURALIST.txt).\n\n"
            "Referencia del laboratorio: revistas.unicomfacauca.edu.co · artículo 507.\n\n"
            "Material de clase: «Introducción a las Redes Neuronales Convolucionales», "
            "Ing. Jaime Andrade, Universidad de Cundinamarca.\n\n"
            f"AquaCNN v{APP_VERSION} · Proyecto académico sin fines comerciales."
        )
        c.setWordWrap(True)
        c.setProperty("muted", True)
        c.setStyleSheet("font-size: 11px;")
        credits.body().addWidget(c)
        bottom.addWidget(credits, 1)
        layout.addLayout(bottom)

        shortcuts = Card(padding=16, spacing=6)
        shortcuts.body().addWidget(SectionHeader("Atajos de teclado", ""))
        for keys, desc in (
            ("Ctrl + 1…7", "Navegar entre páginas"),
            ("Ctrl + O", "Seleccionar carpeta del dataset"),
            ("F5", "Recalcular pipeline"),
            ("Ctrl + E", "Exportar dataset procesado"),
            ("F11", "Alternar pantalla completa"),
            ("Esc", "Volver a la ventana maximizada"),
        ):
            row = QHBoxLayout()
            k = QLabel(keys)
            k.setStyleSheet(
                f"background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px;"
                "padding: 3px 8px; font-family: Consolas; font-size: 10px;"
            )
            row.addWidget(k)
            d = QLabel(desc)
            d.setProperty("muted", True)
            d.setStyleSheet("font-size: 11px;")
            row.addWidget(d, 1)
            shortcuts.body().addLayout(row)
        layout.addWidget(shortcuts)
        layout.addStretch(1)
