"""Iconos SVG vectoriales generados en tiempo de ejecución (sin assets externos)."""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_PATHS: dict[str, str] = {
    "home": '<path d="M3 10.7 12 3l9 7.7V21a1 1 0 0 1-1 1h-5.2v-7H9.2v7H4a1 1 0 0 1-1-1z"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5"/><path d="m3 16.5 9 5 9-5"/>',
    "brain": '<path d="M9.5 3.5a3 3 0 0 0-3 3 3 3 0 0 0-1.6 5.5A3 3 0 0 0 6 17a3 3 0 0 0 3.5 3.2V3.5z"/><path d="M14.5 3.5a3 3 0 0 1 3 3 3 3 0 0 1 1.6 5.5A3 3 0 0 1 18 17a3 3 0 0 1-3.5 3.2V3.5z"/><path d="M12 3.5v17"/>',
    "chart": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.5h.01"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "upload": '<path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "play": '<path d="m7 4 13 8-13 8z"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "save": '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h8V3"/><path d="M8 21v-7h8v7"/>',
    "export": '<path d="M12 4v12"/><path d="m7 11 5 5 5-5"/><path d="M4 19v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1"/>',
    "refresh": '<path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M5.5 8.5A7.5 7.5 0 0 1 19 7l1 5"/><path d="M18.5 15.5A7.5 7.5 0 0 1 5 17l-1-5"/>',
    "sliders": '<path d="M4 6h10"/><path d="M18 6h2"/><circle cx="16" cy="6" r="2"/><path d="M4 12h4"/><path d="M12 12h8"/><circle cx="10" cy="12" r="2"/><path d="M4 18h12"/><path d="M20 18h0"/><circle cx="18" cy="18" r="2"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="m21 15-4.5-4.5L7 20"/>',
    "fish": '<path d="M2.5 12c3.5-5 8-7 13-5.5 2.2.7 3.8 2 5 3.5-1.2 1.5-2.8 2.8-5 3.5-5 1.5-9.5-.5-13-5.5z"/><path d="M2.5 12c1.2.8 2 2 2.5 3.5L2.5 18z"/><circle cx="16" cy="10.5" r=".8" fill="currentColor" stroke="none"/><path d="M9 8.5c1.8 2 1.8 5 0 7"/>',
    "check": '<path d="m4 12.5 5 5L20 6"/>',
    "x": '<path d="M6 6l12 12"/><path d="M18 6 6 18"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "sparkles": '<path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 16.5 19.8 19l2.2.8-2.2.8L19 23l-.8-2.4L16 19.8l2.2-.8z"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/><path d="M4 11.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/>',
    "arrow": '<path d="M4 12h16"/><path d="m14 6 6 6-6 6"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    "trash": '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 14h10l1-14"/><path d="M10 11v6"/><path d="M14 11v6"/>',
    "plus": '<path d="M12 5v14"/><path d="M5 12h14"/>',
    "maximize": '<path d="M4 9V4h5"/><path d="M20 15v5h-5"/><path d="M15 4h5v5"/><path d="M9 20H4v-5"/>',
    "chevron": '<path d="m9 6 6 6-6 6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "cpu": '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "wave": '<path d="M2 12c2.5 0 2.5-6 5-6s2.5 12 5 12 2.5-6 5-6 2.5 3 5 3"/>',
    "lock": '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    "arrow-left": '<path d="M20 12H4"/><path d="m10 6-6 6 6 6"/>',
    "download": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 20h16"/>',
}

_CACHE: dict[tuple[str, str, int], QIcon] = {}
_PIXMAP_CACHE: dict[tuple[str, str, int], QPixmap] = {}


LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0e3b52"/>
      <stop offset="0.55" stop-color="#0a2135"/>
      <stop offset="1" stop-color="#050d17"/>
    </linearGradient>
    <linearGradient id="fish" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#a5f3fc"/>
      <stop offset="0.35" stop-color="#22d3ee"/>
      <stop offset="1" stop-color="#0d9488"/>
    </linearGradient>
    <linearGradient id="rim" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#67e8f9" stop-opacity="0.9"/>
      <stop offset="1" stop-color="#0e7490" stop-opacity="0.25"/>
    </linearGradient>
    <radialGradient id="halo" cx="0.5" cy="0.42" r="0.55">
      <stop offset="0" stop-color="#22d3ee" stop-opacity="0.22"/>
      <stop offset="1" stop-color="#22d3ee" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="6" y="6" width="244" height="244" rx="58" fill="url(#bg)"/>
  <rect x="6" y="6" width="244" height="244" rx="58" fill="none" stroke="url(#rim)" stroke-width="3"/>
  <circle cx="128" cy="112" r="92" fill="url(#halo)"/>
  <path d="M56 126 C78 90 120 76 160 86 C190 94 208 110 218 127 C208 144 190 160 160 168 C120 178 78 164 56 126 Z" fill="url(#fish)"/>
  <path d="M56 126 L28 98 L37 126 L28 154 Z" fill="#22d3ee"/>
  <path d="M64 126 C84 100 116 90 146 96" fill="none" stroke="#ecfeff" stroke-opacity="0.45" stroke-width="5" stroke-linecap="round"/>
  <path d="M150 98 C142 112 142 142 150 156" fill="none" stroke="#0f766e" stroke-opacity="0.55" stroke-width="5" stroke-linecap="round"/>
  <g stroke="#06222e" stroke-width="3.2" opacity="0.9" fill="none">
    <path d="M92 124 L124 108 L156 124 L124 142 Z"/>
    <path d="M124 108 L124 142"/>
    <path d="M92 124 L124 142"/>
    <path d="M156 124 L124 142"/>
  </g>
  <g fill="#06222e">
    <circle cx="92" cy="124" r="7.5"/><circle cx="124" cy="108" r="7.5"/>
    <circle cx="156" cy="124" r="7.5"/><circle cx="124" cy="142" r="7.5"/>
  </g>
  <g fill="#cffafe">
    <circle cx="92" cy="124" r="2.7"/><circle cx="124" cy="108" r="2.7"/>
    <circle cx="156" cy="124" r="2.7"/><circle cx="124" cy="142" r="2.7"/>
  </g>
  <circle cx="184" cy="116" r="8" fill="#04121d"/>
  <circle cx="186.5" cy="113" r="2.8" fill="#ecfeff"/>
  <path d="M44 198 q18 -13 36 0 t36 0 t36 0 t36 0 t24 0" fill="none" stroke="#22d3ee" stroke-opacity="0.38" stroke-width="5" stroke-linecap="round"/>
  <path d="M60 216 q18 -11 36 0 t36 0 t36 0 t36 0" fill="none" stroke="#2dd4bf" stroke-opacity="0.24" stroke-width="4" stroke-linecap="round"/>
</svg>"""


def logo_markup() -> str:
    """Logotipo oficial de AquaCNN (SVG vectorial)."""
    return LOGO_SVG


def logo_pixmap(size: int = 64, dpr: float = 2.0) -> QPixmap:
    """Renderiza el logotipo con supersampling para pantallas HiDPI."""
    px = max(8, int(size * dpr))
    renderer = QSvgRenderer(QByteArray(LOGO_SVG.encode("utf-8")))
    pm = QPixmap(QSize(px, px))
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def svg_markup(name: str, color: str = "#e8f1ff") -> str:
    inner = _PATHS.get(name, _PATHS["fish"])
    inner = inner.replace("currentColor", color)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        f"{inner}</svg>"
    )


def icon(name: str, color: str = "#e8f1ff", size: int = 24) -> QIcon:
    key = (name, color, size)
    if key in _CACHE:
        return _CACHE[key]
    renderer = QSvgRenderer(QByteArray(svg_markup(name, color).encode("utf-8")))
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    ico = QIcon(pm)
    _CACHE[key] = ico
    return ico


def pixmap(name: str, color: str = "#e8f1ff", size: int = 64) -> QPixmap:
    key = (name, color, int(size))
    hit = _PIXMAP_CACHE.get(key)
    if hit is not None:
        return hit
    renderer = QSvgRenderer(QByteArray(svg_markup(name, color).encode("utf-8")))
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    _PIXMAP_CACHE[key] = pm
    return pm


def asset_pixmap(name: str, height: int = 32, dpr: float = 2.0) -> QPixmap:
    """Carga un recurso raster de ``ui/assets`` escalado con suavidad (HiDPI)."""
    try:
        from utils.paths import resource_path

        path = resource_path("ui", "assets", name)
        pm = QPixmap(str(path))
    except Exception:  # noqa: BLE001
        return QPixmap()
    if pm.isNull():
        return pm
    scaled = pm.scaledToHeight(max(1, int(height * dpr)), Qt.TransformationMode.SmoothTransformation)
    scaled.setDevicePixelRatio(dpr)
    return scaled


def app_icon() -> QIcon:
    """Icono oficial (.ico/.png); si no existe, se dibuja el pez vectorial."""
    try:
        from utils.paths import resource_path

        for name in ("aquacnn.ico", "aquacnn.png"):
            path = resource_path("ui", "assets", name)
            if path.exists():
                ico = QIcon(str(path))
                if not ico.isNull():
                    return ico
    except Exception:  # noqa: BLE001
        pass
    from ui.theme import ACCENT

    return icon("fish", ACCENT, 128)
