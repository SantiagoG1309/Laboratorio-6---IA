"""Persistencia de modelos entrenados en formato .npz portable."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.cnn import build_model
from core.config import CLASSES, TrainingConfig

FORMAT_VERSION = 2


def save_model(path: str | Path, model, cfg: TrainingConfig, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    params = model.params()
    arrays = {f"p{i}": p for i, (_, p) in enumerate(params)}
    meta = {
        "format_version": FORMAT_VERSION,
        "model_type": "cnn",
        "input_size": int(cfg.input_size),
        "hidden": int(cfg.hidden_units),
        "filters1": int(cfg.conv_filters),
        "filters2": int(cfg.conv_filters2),
        "n_out": len(CLASSES),
        "dropout": float(cfg.dropout),
        "label_smoothing": float(cfg.label_smoothing),
        "param_names": [name for name, _ in params],
        "param_shapes": [list(p.shape) for _, p in params],
        "training_config": cfg.to_dict(),
        "extra": extra or {},
    }
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    return path


def load_model(path: str | Path):
    """Carga un modelo .npz validando arquitectura, nombres y formas."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data:
            raise ValueError("El archivo no contiene metadatos de AquaCNN")
        meta = json.loads(str(data["meta"]))
        cfg = TrainingConfig.from_dict(meta.get("training_config", {}))
        model = build_model(
            meta.get("model_type", cfg.model_type),
            int(meta.get("input_size", cfg.input_size)),
            int(meta.get("hidden", cfg.hidden_units)),
            int(meta.get("filters1", cfg.conv_filters)),
            int(meta.get("filters2", cfg.conv_filters2)),
            n_out=int(meta.get("n_out", len(CLASSES))),
            seed=cfg.seed,
            dropout=float(meta.get("dropout", cfg.dropout)),
        )
        params = model.params()
        expected_names = [name for name, _ in params]
        saved_names = list(meta.get("param_names") or [])
        if saved_names and saved_names != expected_names:
            raise ValueError("La arquitectura del modelo no coincide con esta versión de AquaCNN")
        saved_shapes = list(meta.get("param_shapes") or [])
        for i, (name, p) in enumerate(params):
            key = f"p{i}"
            if key not in data:
                raise ValueError(f"Modelo incompleto: falta el tensor {key} ({name})")
            arr = data[key]
            if tuple(arr.shape) != tuple(p.shape):
                raise ValueError(
                    f"Forma incompatible en {key} ({name}): esperada {p.shape}, archivo {arr.shape}"
                )
            if saved_shapes and tuple(saved_shapes[i]) != tuple(p.shape):
                raise ValueError(f"Metadatos incoherentes para {key} ({name})")
            p[...] = arr
    return model, meta, cfg
