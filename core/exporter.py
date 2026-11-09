"""Exportación del dataset procesado por etapas."""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np

from core import image_ops as ops
from core.config import CLASS_LABELS, CLASSES, PipelineConfig
from core.dataset import Sample
from core.pipeline import analyze_image

STAGE_FOLDERS = {
    "original": "01_original",
    "canales": "02_canales_rgb_hsv",
    "gris": "03_escala_grises",
    "roi": "04_roi_redimensionada",
    "ecualizado": "05_ecualizado",
    "convolucion": "06_convolucion",
    "pooling": "07_pooling",
}


def export_processed_dataset(
    samples: list[Sample],
    cfg: PipelineConfig,
    output_dir: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Exporta cada etapa del pipeline y un CSV/NPZ de características."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or (lambda _s: None)
    for cls in CLASSES:
        for folder in STAGE_FOLDERS.values():
            (out / cls / folder).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    total = len(samples)
    t0 = time.perf_counter()
    used_stems: dict[str, int] = {}

    def unique_stem(name: str) -> str:
        base = Path(name).stem[:80] or "imagen"
        count = used_stems.get(base, 0) + 1
        used_stems[base] = count
        return base if count == 1 else f"{base}_{count}"

    for i, sample in enumerate(samples, start=1):
        if should_stop and should_stop():
            log("Exportación cancelada.")
            break
        try:
            res = analyze_image(sample.path, cfg)
        except Exception as exc:  # noqa: BLE001
            log(f"  ! {sample.name}: {exc}")
            continue
        base = out / sample.class_name
        stem = unique_stem(sample.name)
        # Se conserva el archivo original sin recomprimir cuando es posible.
        src = Path(sample.path)
        try:
            shutil.copy2(src, base / STAGE_FOLDERS["original"] / f"{stem}{src.suffix.lower()}")
        except OSError:
            ops.imwrite_rgb(base / STAGE_FOLDERS["original"] / f"{stem}.png", res.original)
        ops.imwrite_rgb(base / STAGE_FOLDERS["canales"] / f"{stem}.png", res.channels)
        ops.imwrite_gray(base / STAGE_FOLDERS["gris"] / f"{stem}.png", res.gray)
        ops.imwrite_rgb(base / STAGE_FOLDERS["roi"] / f"{stem}.png", res.work_rgb)
        eq = res.processed_u8
        if eq.ndim == 3:
            eq = ops.to_gray(eq)
        ops.imwrite_gray(base / STAGE_FOLDERS["ecualizado"] / f"{stem}.png", eq)
        ops.imwrite_rgb(base / STAGE_FOLDERS["convolucion"] / f"{stem}.png", res.conv_display)
        ops.imwrite_rgb(base / STAGE_FOLDERS["pooling"] / f"{stem}.png", res.pooled_display)

        vec = res.flat
        vectors.append(vec.astype(np.float32))
        labels.append(sample.label)
        rows.append({
            "archivo": sample.name,
            "clase": sample.class_name,
            "etiqueta": sample.label,
            "ancho_original": res.original.shape[1],
            "alto_original": res.original.shape[0],
            "roi_x": res.roi_rect[0],
            "roi_y": res.roi_rect[1],
            "roi_w": res.roi_rect[2],
            "roi_h": res.roi_rect[3],
            "caracteristicas": int(vec.size),
            "media": float(np.mean(vec)) if vec.size else 0.0,
            "std": float(np.std(vec)) if vec.size else 0.0,
            "min": float(np.min(vec)) if vec.size else 0.0,
            "max": float(np.max(vec)) if vec.size else 0.0,
        })
        if progress:
            progress(i, total, sample.name)
        if i % 10 == 0:
            log(f"  exportadas {i}/{total} imágenes…")

    if rows:
        with open(out / "caracteristicas.csv", "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if vectors:
        max_len = max(v.size for v in vectors)
        matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
        for i, v in enumerate(vectors):
            matrix[i, :v.size] = v
        np.savez_compressed(out / "caracteristicas.npz", X=matrix, y=np.array(labels, dtype=np.int64))
    meta = {
        "creado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_imagenes": len(rows),
        "imagenes_omitidas": max(0, total - len(rows)),
        "clases": {c: CLASS_LABELS[c] for c in CLASSES},
        "pipeline": cfg.to_dict(),
        "etapas": STAGE_FOLDERS,
        "duracion_seg": round(time.perf_counter() - t0, 2),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "LEEME.txt").write_text(
        "Dataset procesado por AquaCNN · Laboratorio 6\n\n"
        "Cada clase contiene las etapas del pipeline:\n"
        + "\n".join(f"  {v}: {k}" for k, v in STAGE_FOLDERS.items())
        + "\n\ncaracteristicas.csv: estadísticas por imagen.\n"
        "caracteristicas.npz: vectores completos (X) y etiquetas (y).\n"
        "metadata.json: configuración exacta del pipeline.\n",
        encoding="utf-8",
    )
    log(f"Exportación finalizada en {out}")
    return out
