"""Genera el logotipo de AquaCNN (PNG + ICO multi-tamaño) desde el SVG interno."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from ui import icons

OUT = ROOT / "ui" / "assets"
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def render(size: int) -> QImage:
    """Renderiza el SVG con 4× de supersampling y reduce con suavidad."""
    scale = 4
    px = size * scale
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer = QSvgRenderer(QByteArray(icons.logo_markup().encode("utf-8")))
    renderer.render(p, QRectF(0, 0, px, px))
    p.end()
    return img.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


def write_ico(path: Path, images: list[QImage]) -> None:
    """Escribe un ICO con entradas PNG (soportado desde Windows Vista)."""
    blobs: list[bytes] = []
    for im in images:
        tmp = OUT / f"_tmp_{im.width()}.png"
        im.save(str(tmp), "PNG")
        blobs.append(tmp.read_bytes())
        tmp.unlink()

    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = bytearray()
    for im, blob in zip(images, blobs):
        w = 0 if im.width() >= 256 else im.width()
        h = 0 if im.height() >= 256 else im.height()
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    path.write_bytes(header + bytes(entries) + b"".join(blobs))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    OUT.mkdir(parents=True, exist_ok=True)
    images = [render(s) for s in SIZES]
    big = render(512)
    big.save(str(OUT / "aquacnn.png"), "PNG")
    write_ico(OUT / "aquacnn.ico", images)
    # Vista previa para revisión visual.
    preview = QPixmap(560, 220)
    preview.fill(Qt.GlobalColor.transparent)
    painter = QPainter(preview)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    x = 14
    for size in (16, 24, 32, 48, 64, 128):
        pm = QPixmap.fromImage(render(size))
        painter.drawPixmap(x, 20, pm)
        x += size + 22
    painter.end()
    preview.save(str(OUT / "logo_preview.png"), "PNG")
    print("PNG:", OUT / "aquacnn.png")
    print("ICO:", OUT / "aquacnn.ico", [im.width() for im in images])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
