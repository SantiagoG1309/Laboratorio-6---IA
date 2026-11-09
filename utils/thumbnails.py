"""Miniaturas en caché para listas y tarjetas de la interfaz."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from core import image_ops as ops
from utils.paths import cache_dir


def thumb_path(source: str | Path, width: int = 240, height: int = 180) -> Path:
    src = Path(source)
    try:
        stamp = f"{src.resolve()}|{src.stat().st_mtime_ns}|{src.stat().st_size}|{width}x{height}"
    except OSError:
        stamp = f"{src}|{width}x{height}"
    digest = hashlib.md5(stamp.encode("utf-8")).hexdigest()[:16]
    return cache_dir() / "thumbs" / f"{digest}.jpg"


def ensure_thumb(source: str | Path, width: int = 240, height: int = 180) -> Path | None:
    """Genera (si hace falta) y devuelve la ruta de la miniatura."""
    src = Path(source)
    dest = thumb_path(src, width, height)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        rgb = ops.imread_rgb(src, reduce=2)
        # Si la imagen original era pequeña, la decodificación reducida no
        # alcanza para la miniatura: se relee a resolución completa.
        if rgb.shape[1] < width or rgb.shape[0] < height:
            rgb = ops.imread_rgb(src)
    except Exception:  # noqa: BLE001
        return None
    h, w = rgb.shape[:2]
    scale = max(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - width) // 2)
    y0 = max(0, (nh - height) // 2)
    crop = resized[y0:y0 + height, x0:x0 + width]
    if crop.shape[0] < height or crop.shape[1] < width:
        canvas = np.full((height, width, 3), 17, np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    ops.imwrite_rgb(dest, crop)
    return dest


def ensure_thumbs(paths: list[str | Path], width: int = 240, height: int = 180) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        tp = ensure_thumb(p, width, height)
        if tp:
            out[str(p)] = str(tp)
    return out
