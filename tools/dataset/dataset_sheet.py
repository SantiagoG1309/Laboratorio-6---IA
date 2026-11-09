"""Hoja de contacto rápida con OpenCV para revisar el dataset."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"
OUT = ROOT / "tools" / "dataset_sheet.png"

files = []
for cls in ("trucha", "mojarra", "pavon"):
    for p in sorted((DS / cls).iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            files.append((cls, p))

cols = 8
cw, ch = 180, 140
rows = math.ceil(len(files) / cols)
sheet = np.full((rows * (ch + 26) + 10, cols * (cw + 8) + 8, 3), 18, np.uint8)
for i, (cls, p) in enumerate(files):
    img = cv2.imread(str(p))
    if img is None:
        continue
    scale = min(cw / img.shape[1], ch / img.shape[0])
    nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
    thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    r, c = divmod(i, cols)
    x, y = 8 + c * (cw + 8), 8 + r * (ch + 26)
    sheet[y:y + nh, x:x + nw] = thumb
    cv2.putText(sheet, f"{i:02d} {p.name[:24]}", (x, y + ch + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 230, 240), 1, cv2.LINE_AA)
cv2.imwrite(str(OUT), sheet)
print("ok", OUT, sheet.shape)
