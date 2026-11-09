"""Tema visual de AquaCNN: paletas, acentos y hoja de estilos.

La paleta se puede cambiar en caliente desde *Configuración*. Para que los
colores lleguen a los estilos en línea de las páginas, ``configure_palette``
actualiza las constantes del módulo antes de reconstruir la interfaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from ui import preferences

# Recursos del tema
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CHEVRON_DOWN = (ASSETS_DIR / "chevron_down.svg").as_posix()
CHEVRON_UP = (ASSETS_DIR / "chevron_up.svg").as_posix()
CHEVRON_RIGHT = (ASSETS_DIR / "chevron_right.svg").as_posix()


@dataclass(frozen=True)
class Theme:
    key: str
    label: str
    hint: str
    bg: str
    bg_alt: str
    surface: str
    surface_2: str
    surface_3: str
    border: str
    border_soft: str
    text: str
    muted: str
    muted_2: str
    hero_a: str
    hero_b: str
    hero_c: str
    sidebar_a: str
    sidebar_b: str
    success: str = "#34d399"
    warning: str = "#fbbf24"
    danger: str = "#fb7185"
    purple: str = "#a78bfa"


THEMES: dict[str, Theme] = {
    "ocean": Theme(
        key="ocean", label="Océano profundo", hint="Azul marino con cian luminoso",
        bg="#070d16", bg_alt="#0a1220", surface="#0d1828", surface_2="#111f33",
        surface_3="#152740", border="#1c314d", border_soft="#16273e",
        text="#e9f2ff", muted="#8ea6c4", muted_2="#617a99",
        hero_a="#0b2a3f", hero_b="#0b2236", hero_c="#101a33",
        sidebar_a="#08111e", sidebar_b="#060b14",
    ),
    "abyss": Theme(
        key="abyss", label="Abismo", hint="Negro azulado de alto contraste",
        bg="#03060b", bg_alt="#060b13", surface="#080f1b", surface_2="#0c1624",
        surface_3="#112033", border="#17293f", border_soft="#101d2e",
        text="#f2f7ff", muted="#93a8c4", muted_2="#5d7391",
        hero_a="#08202f", hero_b="#071827", hero_c="#0b1220",
        sidebar_a="#040910", sidebar_b="#020509",
    ),
    "aurora": Theme(
        key="aurora", label="Aurora", hint="Violeta nocturno con destellos rosas",
        bg="#0a0714", bg_alt="#0e0a1c", surface="#130f24", surface_2="#191330",
        surface_3="#211a3d", border="#2c2450", border_soft="#211b3d",
        text="#f1ecff", muted="#a99cc9", muted_2="#756a97",
        hero_a="#241448", hero_b="#1a1036", hero_c="#140f2b",
        sidebar_a="#0c0818", sidebar_b="#07040f",
        success="#4ade80", warning="#fbbf24", danger="#fb7185", purple="#c4b5fd",
    ),
    "forest": Theme(
        key="forest", label="Bosque", hint="Verde profundo con turquesa",
        bg="#05100d", bg_alt="#071713", surface="#0a1d18", surface_2="#0d261f",
        surface_3="#123329", border="#1b4436", border_soft="#14362b",
        text="#e9fff7", muted="#8fb9a8", muted_2="#5f8a79",
        hero_a="#0c342b", hero_b="#0a2a23", hero_c="#0d211c",
        sidebar_a="#06130f", sidebar_b="#030b08",
        success="#34d399", warning="#fcd34d", danger="#fb7185", purple="#5eead4",
    ),
    "graphite": Theme(
        key="graphite", label="Grafito", hint="Gris neutro con acento personalizable",
        bg="#0b0c0f", bg_alt="#101216", surface="#15181e", surface_2="#1b1f27",
        surface_3="#232833", border="#2c3341", border_soft="#222835",
        text="#f0f3f8", muted="#a3adbd", muted_2="#6d7788",
        hero_a="#1d2430", hero_b="#171d27", hero_c="#14181f",
        sidebar_a="#0d0f13", sidebar_b="#08090c",
        success="#4ade80", warning="#fbbf24", danger="#f87171", purple="#a5b4fc",
    ),
}

ACCENTS: dict[str, tuple[str, str, str, str]] = {
    # clave: (etiqueta, acento, acento secundario, acento oscuro)
    "cian": ("Cian", "#22d3ee", "#2dd4bf", "#0e7490"),
    "azul": ("Azul", "#60a5fa", "#38bdf8", "#1d4ed8"),
    "turquesa": ("Turquesa", "#2dd4bf", "#5eead4", "#0f766e"),
    "violeta": ("Violeta", "#a78bfa", "#c4b5fd", "#6d28d9"),
    "ambar": ("Ámbar", "#fbbf24", "#fcd34d", "#b45309"),
    "rosa": ("Rosa", "#f472b6", "#f9a8d4", "#be185d"),
    "verde": ("Verde", "#34d399", "#6ee7b7", "#047857"),
}

# ---------------------------------------------------------------------------
# Tipografía configurable
# ---------------------------------------------------------------------------
def _pref_font_family() -> str:
    family = str(preferences.get("font_family", "") or "").strip()
    return family or "Segoe UI"


def _pref_font_size() -> int:
    try:
        size = int(preferences.get("font_size", 13))
    except (TypeError, ValueError):
        size = 13
    return max(10, min(20, size))


FONT_FAMILY = _pref_font_family()
FONT_SIZE = _pref_font_size()
FONT_STACK = f'"{FONT_FAMILY}", "Segoe UI", "Inter", "Noto Sans", sans-serif'
FONT_DISPLAY = f'"{FONT_FAMILY}", "Segoe UI Semibold", "Segoe UI", sans-serif'
MONO_STACK = '"Cascadia Code", "Consolas", monospace'


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) == 8:
        value = value[2:]
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def mix(color_a: str, color_b: str, t: float) -> str:
    """Mezcla lineal entre dos colores (0 = color_a, 1 = color_b)."""
    ra, ga, ba = _rgb(color_a)
    rb, gb, bb = _rgb(color_b)
    t = max(0.0, min(1.0, t))
    return "#{:02x}{:02x}{:02x}".format(
        int(round(ra + (rb - ra) * t)),
        int(round(ga + (gb - ga) * t)),
        int(round(ba + (bb - ba) * t)),
    )


def luminance(color: str) -> float:
    r, g, b = (c / 255.0 for c in _rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_ink(background: str) -> str:
    """Texto/icono legible sobre un color de fondo."""
    return "#06121a" if luminance(background) > 0.55 else "#ffffff"


def qcolor(color: str, alpha: int = 255) -> QColor:
    """QColor del tema con alfa opcional (0–255)."""
    value = QColor(color)
    value.setAlpha(max(0, min(255, int(alpha))))
    return value


# Traduce los colores semánticos fijos de las etapas a la paleta activa.
_STAGE_ALIASES: dict[str, str] = {
    "#38bdf8": "accent_2",
    "#22d3ee": "accent",
    "#2dd4bf": "accent_mix",
    "#a78bfa": "purple",
    "#f472b6": "accent_purple",
    "#fbbf24": "warning",
    "#34d399": "success",
    "#fb7185": "danger",
}


def stage_color(color: str) -> str:
    """Devuelve el color de una etapa adaptado al tema y acento activos."""
    key = _STAGE_ALIASES.get(str(color).lower())
    if key is None:
        return color
    return {
        "accent": ACCENT,
        "accent_2": ACCENT_2,
        "accent_mix": mix(ACCENT, ACCENT_2, 0.5),
        "purple": PURPLE,
        "accent_purple": mix(ACCENT, PURPLE, 0.55),
        "warning": WARNING,
        "success": SUCCESS,
        "danger": DANGER,
    }[key]


# ---------------------------------------------------------------------------
# Constantes de la paleta activa (se reasignan con configure_palette)
# ---------------------------------------------------------------------------
BG = "#070d16"
BG_ALT = "#0a1220"
SURFACE = "#0d1828"
SURFACE_2 = "#111f33"
SURFACE_3 = "#152740"
BORDER = "#1c314d"
BORDER_SOFT = "#16273e"
TEXT = "#e9f2ff"
MUTED = "#8ea6c4"
MUTED_2 = "#617a99"
ACCENT = "#22d3ee"
ACCENT_2 = "#2dd4bf"
ACCENT_DARK = "#0e7490"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#fb7185"
PURPLE = "#a78bfa"
HERO_A = "#0b2a3f"
HERO_B = "#0b2236"
HERO_C = "#101a33"
SIDEBAR_A = "#08111e"
SIDEBAR_B = "#060b14"

# Colores derivados (se recalculan con la paleta activa)
ON_ACCENT = "#06121a"
TOGGLE_OFF = "#1b2d45"
KNOB_ON = "#f8fdff"
KNOB_OFF = "#7f95b2"
CHART_GRID = "#1a2d45"
PANEL = "#08111d"
DANGER_TEXT = "#ffe4e6"
SUCCESS_TEXT = "#d1fae5"

CURRENT_THEME = "ocean"
CURRENT_ACCENT = "cian"


def rgba(color: str, alpha: float) -> str:
    """Convierte «#rrggbb» + alfa (0–1) en «rgba(...)» para QSS."""
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) == 8:  # ya venía con alfa: se ignora
        value = value[2:]
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, alpha)):.3f})"


def configure_palette(theme_key: str | None = None, accent_key: str | None = None) -> None:
    """Actualiza las constantes del módulo con la paleta seleccionada."""
    global BG, BG_ALT, SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT
    global TEXT, MUTED, MUTED_2, ACCENT, ACCENT_2, ACCENT_DARK
    global SUCCESS, WARNING, DANGER, PURPLE
    global HERO_A, HERO_B, HERO_C, SIDEBAR_A, SIDEBAR_B
    global ON_ACCENT, TOGGLE_OFF, KNOB_ON, KNOB_OFF, CHART_GRID, PANEL
    global DANGER_TEXT, SUCCESS_TEXT
    global FONT_FAMILY, FONT_SIZE, FONT_STACK, FONT_DISPLAY
    global CURRENT_THEME, CURRENT_ACCENT, QSS

    theme_key = theme_key if theme_key in THEMES else CURRENT_THEME
    accent_key = accent_key if accent_key in ACCENTS else CURRENT_ACCENT
    theme = THEMES[theme_key]
    _label, accent, accent_2, accent_dark = ACCENTS[accent_key]

    BG = theme.bg
    BG_ALT = theme.bg_alt
    SURFACE = theme.surface
    SURFACE_2 = theme.surface_2
    SURFACE_3 = theme.surface_3
    BORDER = theme.border
    BORDER_SOFT = theme.border_soft
    TEXT = theme.text
    MUTED = theme.muted
    MUTED_2 = theme.muted_2
    ACCENT = accent
    ACCENT_2 = accent_2
    ACCENT_DARK = accent_dark
    SUCCESS = theme.success
    WARNING = theme.warning
    DANGER = theme.danger
    PURPLE = theme.purple
    HERO_A = theme.hero_a
    HERO_B = theme.hero_b
    HERO_C = theme.hero_c
    SIDEBAR_A = theme.sidebar_a
    SIDEBAR_B = theme.sidebar_b
    ON_ACCENT = readable_ink(ACCENT)
    TOGGLE_OFF = mix(SURFACE_3, BORDER, 0.35)
    KNOB_ON = mix(TEXT, "#ffffff", 0.55)
    KNOB_OFF = MUTED_2
    CHART_GRID = mix(BORDER, BG, 0.35)
    PANEL = BG_ALT
    DANGER_TEXT = mix(DANGER, "#ffffff", 0.75)
    SUCCESS_TEXT = mix(SUCCESS, "#ffffff", 0.75)
    FONT_FAMILY = _pref_font_family()
    FONT_SIZE = _pref_font_size()
    FONT_STACK = f'"{FONT_FAMILY}", "Segoe UI", "Inter", "Noto Sans", sans-serif'
    FONT_DISPLAY = f'"{FONT_FAMILY}", "Segoe UI Semibold", "Segoe UI", sans-serif'
    CURRENT_THEME = theme_key
    CURRENT_ACCENT = accent_key
    QSS = build_qss()


def load_saved_palette() -> tuple[str, str]:
    theme = str(preferences.get("theme", "ocean"))
    accent = str(preferences.get("accent", "cian"))
    if theme not in THEMES:
        theme = "ocean"
    if accent not in ACCENTS:
        accent = "cian"
    configure_palette(theme, accent)
    return theme, accent


def build_qss() -> str:
    """Hoja de estilos completa con la paleta activa."""
    fs = FONT_SIZE
    small = max(9, fs - 2)
    tiny = max(8, fs - 3)
    return f"""
* {{
    outline: none;
}}
QWidget {{
    color: {TEXT};
    font-family: {FONT_STACK};
    font-size: {fs}px;
}}
QMainWindow, QWidget#Root, QWidget#Page {{
    background: {BG};
}}
QToolTip {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {ACCENT_DARK};
    border-radius: 8px;
    padding: 7px 10px;
    font-size: {small}px;
}}
QFrame#Sidebar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {SIDEBAR_A}, stop:0.55 {SIDEBAR_A}, stop:1 {SIDEBAR_B});
    border-right: 1px solid {BORDER_SOFT};
}}
QFrame#TopBar {{
    background: transparent;
    border-bottom: 1px solid {BORDER_SOFT};
}}
QFrame[card="true"] {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {SURFACE}, stop:1 {BG_ALT});
    border: 1px solid {BORDER};
    border-radius: 16px;
}}
QFrame[card="true"]:hover {{
    border: 1px solid {BORDER};
}}
QFrame[soft="true"] {{
    background: {SURFACE_2};
    border: 1px solid {BORDER_SOFT};
    border-radius: 12px;
}}
QFrame#Hero {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {HERO_A}, stop:0.5 {HERO_B}, stop:1 {HERO_C});
    border: 1px solid {BORDER};
    border-radius: 20px;
}}
QFrame#StageCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {SURFACE_2}, stop:1 {BG_ALT});
    border: 1px solid {BORDER};
    border-radius: 14px;
}}
QFrame#StageCard:hover {{
    border: 1px solid {ACCENT_DARK};
}}
QLabel {{
    background: transparent;
}}
QLabel[h1="true"] {{
    font-family: {FONT_DISPLAY};
    font-size: {fs + 13}px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel[h2="true"] {{
    font-family: {FONT_DISPLAY};
    font-size: {fs + 5}px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel[h3="true"] {{
    font-family: {FONT_DISPLAY};
    font-size: {fs + 1}px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[muted="true"] {{
    color: {MUTED};
}}
QLabel[subtle="true"] {{
    color: {MUTED_2};
    font-size: {small}px;
}}
QLabel[accent="true"] {{
    color: {ACCENT};
}}
QLabel[mono="true"] {{
    font-family: {MONO_STACK};
    font-size: {small}px;
    color: {MUTED};
}}
QPushButton {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 7px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {SURFACE_3};
    border-color: {ACCENT_DARK};
}}
QPushButton:pressed {{
    background: {BG_ALT};
}}
QPushButton:disabled {{
    color: {MUTED_2};
    background: {BG_ALT};
    border-color: {BORDER_SOFT};
}}
QPushButton[variant="primary"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT_DARK}, stop:1 {ACCENT});
    color: {ON_ACCENT};
    border: 1px solid {ACCENT};
    font-weight: 700;
}}
QPushButton[variant="primary"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ACCENT}, stop:1 {ACCENT_2});
}}
QPushButton[variant="ghost"] {{
    background: transparent;
    border: 1px solid {BORDER};
}}
QPushButton[variant="ghost"]:hover {{
    background: {SURFACE_2};
}}
QPushButton[variant="danger"] {{
    background: {rgba(DANGER, 0.16)};
    color: {DANGER_TEXT};
    border: 1px solid {rgba(DANGER, 0.5)};
}}
QPushButton[variant="danger"]:hover {{
    background: {rgba(DANGER, 0.26)};
}}
QPushButton[variant="success"] {{
    background: {rgba(SUCCESS, 0.16)};
    color: {SUCCESS_TEXT};
    border: 1px solid {rgba(SUCCESS, 0.45)};
}}
QPushButton[variant="nav"] {{
    background: transparent;
    border: none;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: left;
    color: {MUTED};
    font-weight: 500;
}}
QPushButton[variant="nav"]:hover {{
    background: {SURFACE_2};
    color: {TEXT};
}}
QPushButton[variant="nav"]:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {rgba(ACCENT, 0.18)}, stop:1 {rgba(ACCENT, 0.05)});
    color: {TEXT};
    border-left: 3px solid {ACCENT};
}}
QPushButton[variant="chip"] {{
    background: transparent;
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 4px 12px;
    color: {MUTED};
    font-size: {small}px;
}}
QPushButton[variant="chip"]:checked {{
    background: {rgba(ACCENT, 0.16)};
    color: {TEXT};
    border-color: {ACCENT_DARK};
}}
QPushButton[variant="icon"] {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 5px;
}}
QPushButton[variant="icon"]:hover {{
    background: {SURFACE_2};
    border-color: {BORDER};
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {BG_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 9px;
    selection-background-color: {ACCENT_DARK};
    min-height: 18px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {ACCENT_DARK};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 26px;
}}
QComboBox::down-arrow {{
    image: url("{CHEVRON_DOWN}");
    width: 12px;
    height: 12px;
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {ACCENT_DARK};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    background: transparent;
    border: none;
    border-left: 1px solid {BORDER_SOFT};
    border-top-right-radius: 8px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    background: transparent;
    border: none;
    border-left: 1px solid {BORDER_SOFT};
    border-bottom-right-radius: 8px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {SURFACE_3};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{CHEVRON_UP}");
    width: 9px;
    height: 9px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{CHEVRON_DOWN}");
    width: 9px;
    height: 9px;
}}
QCheckBox {{
    spacing: 8px;
    color: {MUTED};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1px solid {BORDER};
    background: {BG_ALT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT_DARK};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 5px;
    background: {SURFACE_3};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_DARK}, stop:1 {ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 15px;
    height: 15px;
    margin: -5px 0;
    border-radius: 8px;
    background: {KNOB_ON};
    border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{
    background: #ffffff;
}}
QProgressBar {{
    background: {BG_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT};
    font-size: {tiny}px;
    min-height: 10px;
    max-height: 12px;
}}
QProgressBar::chunk {{
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_DARK}, stop:1 {ACCENT});
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_DARK};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QListWidget, QTableWidget, QPlainTextEdit, QTextEdit {{
    background: {BG_ALT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    selection-background-color: {ACCENT_DARK};
    selection-color: {TEXT};
}}
QListWidget::item {{
    border-radius: 8px;
    padding: 6px;
    margin: 2px 3px;
    color: {MUTED};
}}
QListWidget::item:hover {{
    background: {SURFACE_2};
    color: {TEXT};
}}
QListWidget::item:selected {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {rgba(ACCENT, 0.2)}, stop:1 {SURFACE_2});
    color: {TEXT};
}}
QTableWidget {{
    gridline-color: {BORDER_SOFT};
    alternate-background-color: {SURFACE};
}}
QHeaderView::section {{
    background: {SURFACE_2};
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 7px;
    font-size: {small}px;
    font-weight: 600;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {SURFACE};
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 7px 14px;
    border-radius: 8px;
    margin-right: 4px;
}}
QTabBar::tab:selected {{
    background: {SURFACE_2};
    color: {TEXT};
}}
QSplitter::handle {{
    background: transparent;
    width: 6px;
    height: 6px;
}}
QSplitter::handle:hover {{
    background: {BORDER_SOFT};
}}
QMenu {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 18px;
    border-radius: 6px;
}}
QMenu::item:selected {{
    background: {ACCENT_DARK};
}}
QDialog {{
    background: {BG_ALT};
}}
QStatusBar {{
    background: {BG_ALT};
    color: {MUTED};
    border-top: 1px solid {BORDER_SOFT};
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    color: {MUTED};
    font-size: {small}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {MUTED};
}}
"""


QSS = build_qss()


def apply_theme(
    app: QApplication,
    theme_key: str | None = None,
    accent_key: str | None = None,
) -> None:
    """Aplica estilo Fusion, paleta de Qt y hoja de estilos."""
    if theme_key or accent_key:
        configure_palette(theme_key, accent_key)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_ALT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_2))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_2))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DARK))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    app.setPalette(palette)
    font = QFont(FONT_FAMILY, max(7, int(round(FONT_SIZE * 0.75))))
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(QSS)


def theme_label(key: str) -> str:
    return THEMES.get(key, THEMES["ocean"]).label


def accent_label(key: str) -> str:
    return ACCENTS.get(key, ACCENTS["cian"])[0]


def accent_color(key: str) -> str:
    return ACCENTS.get(key, ACCENTS["cian"])[1]


__all__ = [
    "ACCENTS", "THEMES", "apply_theme", "configure_palette", "load_saved_palette",
    "rgba", "mix", "luminance", "readable_ink", "qcolor", "stage_color", "theme_label", "accent_label",
    "accent_color", "build_qss",
]
