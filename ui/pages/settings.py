"""Página «Configuración»: apariencia, tipografía, comportamiento y datos."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFontComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import APP_VERSION, MAX_DATASET_IMAGES, UNIVERSITY_FULL
from ui import icons, preferences
from ui.theme import ACCENT, ACCENTS, BORDER, MUTED, ON_ACCENT, THEMES, accent_color, rgba
from ui.widgets.cards import Badge, Card, Divider, SectionHeader, ToggleSwitch


class ThemePreview(QFrame):
    """Muestra los colores del tema y del acento seleccionados."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(96)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(16, 12, 16, 12)
        self._row.setSpacing(10)
        self._title = QLabel("Vista previa del tema")
        self._title.setStyleSheet("font-size: 12px; font-weight: 700; background: transparent;")
        self._row.addWidget(self._title, 1)
        self._dots: list[QLabel] = []
        for _ in range(4):
            dot = QLabel()
            dot.setFixedSize(18, 18)
            self._dots.append(dot)
            self._row.addWidget(dot)

    def set_theme(self, theme_key: str, accent_key: str) -> None:
        theme = THEMES.get(theme_key, THEMES["ocean"])
        accent = accent_color(accent_key)
        self.setStyleSheet(
            f"QFrame {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            f" stop:0 {theme.hero_a}, stop:1 {theme.hero_b});"
            f" border: 1px solid {theme.border}; border-radius: 12px; }}"
            "QLabel { background: transparent; }"
        )
        self._title.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {theme.text}; background: transparent;"
        )
        palette = [accent, theme.text, theme.muted, theme.surface_2]
        for dot, color in zip(self._dots, palette):
            dot.setStyleSheet(
                f"background: {color}; border: 1px solid {rgba(theme.text, 0.25)}; border-radius: 9px;"
            )


class SettingsPage(QWidget):
    """Preferencias persistentes de la aplicación."""

    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent)
        self.main = main
        self._loading = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        content.setObjectName("Page")
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 26)
        layout.setSpacing(14)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel("Configuración")
        title.setProperty("h1", True)
        col.addWidget(title)
        subtitle = QLabel(
            "Apariencia, tipografía, animaciones, tooltips y preferencias del laboratorio. "
            "Los cambios de tema y de fuente se aplican al reiniciar la interfaz."
        )
        subtitle.setProperty("muted", True)
        subtitle.setWordWrap(True)
        col.addWidget(subtitle)
        head.addLayout(col, 1)
        self.version_badge = Badge(f"v{APP_VERSION}", ACCENT)
        head.addWidget(self.version_badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)

        # ---------------- Apariencia ----------------
        look = Card(padding=16, spacing=10)
        look.body().addWidget(SectionHeader("Apariencia", "Tema y color de acento"))
        self.preview = ThemePreview()
        look.body().addWidget(self.preview)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        self.theme_combo = QComboBox()
        for key, theme in THEMES.items():
            self.theme_combo.addItem(f"{theme.label} · {theme.hint}", key)
        self.theme_combo.setToolTip("Cambia la paleta de fondos, tarjetas, textos y bordes.")
        self.accent_combo = QComboBox()
        for key, (label, color, _a2, _ad) in ACCENTS.items():
            self.accent_combo.addItem(label, key)
        self.accent_combo.setToolTip("Color de botones, interruptores, iconos, gráficos y resaltados.")
        grid.addWidget(self._label("Tema de la aplicación", "Cambia fondos, superficies y bordes."), 0, 0)
        grid.addWidget(self.theme_combo, 1, 0)
        grid.addWidget(self._label("Color de acento", "Afecta botones, iconos, gráficos y estados activos."), 0, 1)
        grid.addWidget(self.accent_combo, 1, 1)
        look.body().addLayout(grid)
        apply_row = QHBoxLayout()
        self.apply_theme_btn = QPushButton("Aplicar tema y reiniciar")
        self.apply_theme_btn.setProperty("variant", "primary")
        self.apply_theme_btn.setIcon(icons.icon("refresh", ON_ACCENT, 15))
        self.apply_theme_btn.setToolTip(
            "Guarda el tema y reinicia AquaCNN para que todos los colores se apliquen desde el inicio."
        )
        self.reset_btn = QPushButton("Restablecer preferencias")
        self.reset_btn.setProperty("variant", "ghost")
        self.reset_btn.setIcon(icons.icon("refresh", MUTED, 14))
        apply_row.addWidget(self.apply_theme_btn)
        apply_row.addWidget(self.reset_btn)
        apply_row.addStretch(1)
        look.body().addLayout(apply_row)
        self.theme_note = QLabel(
            "El tema se guarda en las preferencias. Al reiniciar, la interfaz completa "
            "(incluidos los estilos en línea, los iconos y los gráficos) usa la nueva paleta."
        )
        self.theme_note.setProperty("subtle", True)
        self.theme_note.setWordWrap(True)
        look.body().addWidget(self.theme_note)
        layout.addWidget(look)

        # ---------------- Tipografía ----------------
        typo = Card(padding=16, spacing=10)
        typo.body().addWidget(SectionHeader("Tipografía", "Fuente y tamaño del texto"))
        tgrid = QGridLayout()
        tgrid.setHorizontalSpacing(14)
        tgrid.setVerticalSpacing(8)
        self.font_combo = QFontComboBox()
        self.font_combo.setToolTip("Familia tipográfica de toda la interfaz.")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(11, 18)
        self.size_spin.setSuffix(" px")
        self.size_spin.setToolTip("Tamaño base del texto. Los títulos crecen proporcionalmente.")
        tgrid.addWidget(self._label("Fuente", "Se aplica a títulos, botones, tablas y etiquetas."), 0, 0)
        tgrid.addWidget(self.font_combo, 1, 0)
        tgrid.addWidget(self._label("Tamaño base", "De 11 a 18 píxeles."), 0, 1)
        tgrid.addWidget(self.size_spin, 1, 1)
        typo.body().addLayout(tgrid)
        self.font_preview = QLabel("Aa Bb Cc · Trucha, mojarra y pavón · 128 × 128 × 3")
        self.font_preview.setStyleSheet(
            f"font-size: 15px; color: {MUTED}; background: {rgba(ACCENT, 0.06)};"
            f"border: 1px dashed {BORDER}; border-radius: 8px; padding: 10px 12px;"
        )
        typo.body().addWidget(self.font_preview)
        trow = QHBoxLayout()
        self.apply_font_btn = QPushButton("Aplicar tipografía y reiniciar")
        self.apply_font_btn.setProperty("variant", "primary")
        self.apply_font_btn.setIcon(icons.icon("refresh", ON_ACCENT, 15))
        self.apply_font_btn.setToolTip("Guarda la fuente y el tamaño, y reinicia para aplicarlos a todo.")
        trow.addWidget(self.apply_font_btn)
        trow.addStretch(1)
        typo.body().addLayout(trow)
        typo_note = QLabel(
            "La fuente afecta a toda la aplicación, incluidos los gráficos. Si eliges una fuente "
            "que no existe en el sistema, se usa Segoe UI como respaldo."
        )
        typo_note.setProperty("subtle", True)
        typo_note.setWordWrap(True)
        typo.body().addWidget(typo_note)
        layout.addWidget(typo)

        # ---------------- Comportamiento ----------------
        behavior = Card(padding=16, spacing=8)
        behavior.body().addWidget(SectionHeader("Comportamiento", "Opciones de la interfaz y del laboratorio"))
        self.anim_toggle = self._toggle("Animaciones del diagrama y de los interruptores")
        self.tooltips_toggle = self._toggle("Mostrar tooltips explicativos al pasar el cursor")
        self.maximized_toggle = self._toggle("Abrir la aplicación maximizada")
        self.auto_toggle = self._toggle("Actualización automática del pipeline visual")
        for row in (self.anim_toggle, self.tooltips_toggle, self.maximized_toggle, self.auto_toggle):
            behavior.body().addWidget(row)
        behavior.body().addWidget(Divider())
        thumbs_row = QHBoxLayout()
        thumbs_lbl = QLabel("Miniaturas por clase al escanear")
        thumbs_lbl.setProperty("muted", True)
        thumbs_row.addWidget(thumbs_lbl, 1)
        self.thumbs_spin = QSpinBox()
        self.thumbs_spin.setRange(10, 300)
        self.thumbs_spin.setSingleStep(10)
        self.thumbs_spin.setToolTip(
            "Cuántas miniaturas se generan por clase. Más miniaturas embellecen la lista pero tardan más."
        )
        thumbs_row.addWidget(self.thumbs_spin)
        behavior.body().addLayout(thumbs_row)
        layout.addWidget(behavior)

        # ---------------- Datos ----------------
        data = Card(padding=16, spacing=8)
        data.body().addWidget(SectionHeader("Datos y proyecto", "Ruta del dataset y acciones rápidas"))
        self.path_label = QLabel("—")
        self.path_label.setProperty("mono", True)
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet(
            f"background: {rgba(ACCENT, 0.06)}; border: 1px solid {BORDER};"
            "border-radius: 8px; padding: 8px 10px;"
        )
        data.body().addWidget(self.path_label)
        drow = QHBoxLayout()
        self.open_btn = QPushButton("Abrir carpeta")
        self.open_btn.setProperty("variant", "ghost")
        self.open_btn.setIcon(icons.icon("folder", MUTED, 14))
        self.choose_btn = QPushButton("Elegir carpeta")
        self.choose_btn.setProperty("variant", "ghost")
        self.choose_btn.setIcon(icons.icon("upload", MUTED, 14))
        self.rescan_btn = QPushButton("Volver a escanear")
        self.rescan_btn.setProperty("variant", "primary")
        self.rescan_btn.setIcon(icons.icon("refresh", ON_ACCENT, 14))
        for b in (self.open_btn, self.choose_btn, self.rescan_btn):
            drow.addWidget(b)
        drow.addStretch(1)
        data.body().addLayout(drow)
        self.data_note = QLabel(
            f"Tope de seguridad: {MAX_DATASET_IMAGES:,} imágenes en total. ".replace(",", ".")
            + f"{UNIVERSITY_FULL} · Proyecto académico."
        )
        self.data_note.setProperty("subtle", True)
        self.data_note.setWordWrap(True)
        data.body().addWidget(self.data_note)
        layout.addWidget(data)
        layout.addStretch(1)

        # ---------------- Conexiones ----------------
        self.theme_combo.currentIndexChanged.connect(self._update_preview)
        self.accent_combo.currentIndexChanged.connect(self._update_preview)
        self.apply_theme_btn.clicked.connect(self._apply_theme)
        self.reset_btn.clicked.connect(self._reset)
        self.apply_font_btn.clicked.connect(self._apply_font)
        self.font_combo.currentFontChanged.connect(self._update_font_preview)
        self.size_spin.valueChanged.connect(self._update_font_preview)
        for toggle, key in (
            (self.anim_toggle, "animations"),
            (self.tooltips_toggle, "tooltips"),
            (self.maximized_toggle, "start_maximized"),
            (self.auto_toggle, "auto_pipeline"),
        ):
            toggle._switch.toggled.connect(lambda on, k=key: self._save_toggle(k, on))  # type: ignore[attr-defined]
        self.thumbs_spin.valueChanged.connect(lambda v: preferences.set_value("thumbs", int(v)))
        self.open_btn.clicked.connect(self.main.open_dataset_folder)
        self.choose_btn.clicked.connect(self.main.choose_dataset_folder)
        self.rescan_btn.clicked.connect(self.main.rescan_dataset)
        self.refresh()
        self._loading = False

    # ------------------------------------------------------------------
    def _label(self, text: str, tip: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
        lbl.setToolTip(tip)
        return lbl

    def _toggle(self, text: str) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(text)
        lbl.setProperty("muted", True)
        lbl.setStyleSheet("font-size: 11px;")
        rl.addWidget(lbl, 1)
        sw = ToggleSwitch(False)
        rl.addWidget(sw)
        row._switch = sw  # type: ignore[attr-defined]
        return row

    def _update_preview(self) -> None:
        self.preview.set_theme(self.theme_combo.currentData(), self.accent_combo.currentData())

    def _update_font_preview(self) -> None:
        family = self.font_combo.currentFont().family()
        size = int(self.size_spin.value())
        self.font_preview.setStyleSheet(
            f"font-family: '{family}'; font-size: {size + 2}px; color: {MUTED};"
            f"background: {rgba(ACCENT, 0.06)}; border: 1px dashed {BORDER};"
            "border-radius: 8px; padding: 10px 12px;"
        )

    def _save_toggle(self, key: str, value: bool) -> None:
        if self._loading:
            return
        preferences.set_value(key, bool(value))
        if key == "auto_pipeline":
            try:
                self.main.explorer.controls.auto_toggle._switch.setChecked(bool(value))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    def _apply_theme(self) -> None:
        preferences.set_value("theme", self.theme_combo.currentData())
        preferences.set_value("accent", self.accent_combo.currentData())
        self._restart("Guardando tema y reiniciando AquaCNN…")

    def _apply_font(self) -> None:
        family = self.font_combo.currentFont().family()
        preferences.set_value("font_family", "" if family == "Segoe UI" else family)
        preferences.set_value("font_size", int(self.size_spin.value()))
        self._restart("Guardando tipografía y reiniciando AquaCNN…")

    def _restart(self, message: str) -> None:
        if getattr(self.main, "is_training", lambda: False)():
            self.main.toast(
                "Hay un entrenamiento en curso: el cambio se aplicará al terminar y reiniciar.",
                "warning",
            )
            return
        self.main.toast(message, "info")
        QTimer.singleShot(500, self.main.restart_application)

    def _reset(self) -> None:
        preferences.reset()
        self.main.toast("Preferencias restablecidas. Reiniciando…", "info")
        QTimer.singleShot(500, self.main.restart_application)

    def refresh(self) -> None:
        self._loading = True
        try:
            theme = str(preferences.get("theme", "ocean"))
            accent = str(preferences.get("accent", "cian"))
            idx = self.theme_combo.findData(theme)
            self.theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
            idx = self.accent_combo.findData(accent)
            self.accent_combo.setCurrentIndex(idx if idx >= 0 else 0)
            family = str(preferences.get("font_family", "") or "Segoe UI")
            self.font_combo.setCurrentFont(QFont(family))
            self.size_spin.setValue(int(preferences.get("font_size", 13)))
            self.anim_toggle._switch.setChecked(bool(preferences.get("animations", True)))  # type: ignore[attr-defined]
            self.tooltips_toggle._switch.setChecked(bool(preferences.get("tooltips", True)))  # type: ignore[attr-defined]
            self.maximized_toggle._switch.setChecked(bool(preferences.get("start_maximized", True)))  # type: ignore[attr-defined]
            self.auto_toggle._switch.setChecked(bool(preferences.get("auto_pipeline", True)))  # type: ignore[attr-defined]
            self.thumbs_spin.setValue(int(preferences.get("thumbs", 60)))
            self.path_label.setText(str(getattr(self.main, "dataset_path", "—")))
            self._update_preview()
            self._update_font_preview()
        finally:
            self._loading = False
