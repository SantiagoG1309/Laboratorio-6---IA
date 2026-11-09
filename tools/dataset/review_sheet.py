"""Hoja de contacto por clase para revisar visualmente el dataset."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"
OUT = ROOT / "tools" / "dataset" / "sheets"
COLS, ROWS = 7, 6
PER_PAGE = COLS * ROWS
CW, CH = 186, 138


def sheet_for(cls: str, files: list[Path], page: int) -> Path:
    chunk = files[page * PER_PAGE:(page + 1) * PER_PAGE]
    sheet = np.full((ROWS * (CH + 24) + 10, COLS * (CW + 8) + 8, 3), 16, np.uint8)
    for i, p in enumerate(chunk):
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = min(CW / img.shape[1], CH / img.shape[0])
        nw, nh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
        thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, COLS)
        x, y = 8 + c * (CW + 8), 8 + r * (CH + 24)
        sheet[y:y + nh, x:x + nw] = thumb
        label = f"{page * PER_PAGE + i:03d} {p.name[:24]}"
        cv2.putText(sheet, label, (x, y + CH + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (225, 235, 245), 1, cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"review_{cls}_{page + 1}.jpg"
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return dest


def main() -> int:
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or ["trucha", "mojarra", "pavon"]
    for cls in wanted:
        files = sorted(
            p for p in (DS / cls).iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        pages = max(1, math.ceil(len(files) / PER_PAGE))
        for page in range(pages):
            dest = sheet_for(cls, files, page)
            print(cls, len(files), f"p{page + 1}/{pages}", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
