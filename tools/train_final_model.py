"""Entrena el modelo CNN incluido en la aplicación y lo guarda en ``Modelos entrenados``.

Uso:
    .venv\\Scripts\\python.exe tools\\train_final_model.py
    .venv\\Scripts\\python.exe tools\\train_final_model.py --epochs 80 --size 64

El modelo resultante es el que la app carga automáticamente al arrancar.
La división train/val/test es exactamente la misma que usa la interfaz
(70/15/15, semilla 42, estratificada).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import APP_VERSION, CLASS_LABELS, CLASSES, TrainingConfig  # noqa: E402
from core.dataset import scan_dataset, split_dataset  # noqa: E402
from core.model_io import save_model  # noqa: E402
from core.training import train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena y guarda el modelo incluido de AquaCNN")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--size", type=int, default=48, choices=(32, 40, 48, 64))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--filters", type=int, default=6)
    parser.add_argument("--filters2", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Dataset Imagenes")
    parser.add_argument("--output", type=Path, default=ROOT / "Modelos entrenados" / "AquaCNN_cnn_peces.npz")
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()

    cfg = TrainingConfig(
        input_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=args.wd,
        hidden_units=args.hidden,
        conv_filters=args.filters,
        conv_filters2=args.filters2,
        seed=args.seed,
        augmentation=not args.no_augmentation,
    )

    info = scan_dataset(args.dataset)
    if info.is_empty:
        print(f"No hay imágenes en {args.dataset}", file=sys.stderr)
        return 1
    splits = split_dataset(
        info.samples,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        seed=cfg.seed,
        stratified=True,
    )
    counts = " · ".join(f"{CLASS_LABELS[c]}={info.counts.get(c, 0)}" for c in CLASSES)
    print(f"Dataset: {info.total} imágenes ({counts})")
    print(
        f"División: train={len(splits['train'])} · val={len(splits['val'])} · test={len(splits['test'])}"
    )
    print(
        f"CNN: entrada {cfg.input_size}px · filtros {cfg.conv_filters}/{cfg.conv_filters2} · "
        f"ocultas {cfg.hidden_units} · {cfg.epochs} épocas · batch {cfg.batch_size} · lr {cfg.learning_rate}"
    )

    t0 = time.perf_counter()
    last = {"epoch": 0}

    def on_phase(phase: dict) -> None:
        names = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
        name = names.get(str(phase.get("phase", "")), "")
        status = str(phase.get("status", ""))
        if not name:
            return
        if status == "start":
            print(f"Fase: {name}", flush=True)
        elif str(phase.get("phase")) == "val" and status == "progress":
            print(
                f"  instantánea época {int(phase.get('epoch', 0)):3d} · "
                f"val_acc {float(phase.get('val_accuracy', 0.0)) * 100:5.1f}%",
                flush=True,
            )

    def on_epoch(evo: dict) -> None:
        last.update(evo)
        val_acc = evo.get("val_accuracy")
        suffix = "" if val_acc is None else f" · val_acc {float(val_acc) * 100:5.1f}%"
        print(
            f"  época {evo['epoch']:3d}/{evo['epochs']} · loss {evo['loss']:.4f} · "
            f"acc {evo['accuracy'] * 100:5.1f}%{suffix}",
            flush=True,
        )

    result = train_model(
        splits["train"],
        splits["val"],
        splits["test"],
        cfg,
        on_epoch=on_epoch,
        on_phase=on_phase,
        on_log=lambda msg: None,
    )

    extra = {
        "accuracy": float(result.metrics.get("accuracy", 0.0)),
        "macro_f1": float(result.metrics.get("macro_f1", 0.0)),
        "best_val_accuracy": float(result.metrics.get("best_val_accuracy", 0.0)),
        "best_epoch": int(result.best_epoch),
        "duration": float(result.duration),
        "train_count": int(result.train_count),
        "val_count": int(result.val_count),
        "test_count": int(result.test_count),
        "dataset_counts": {c: int(info.counts.get(c, 0)) for c in CLASSES},
        "history": {k: [float(v) for v in vals] for k, vals in result.history.items()},
        "app_version": APP_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_seed": int(cfg.seed),
        "split_ratios": [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio],
    }
    saved = save_model(args.output, result.model, cfg, extra)
    elapsed = time.perf_counter() - t0
    print()
    print(f"Exactitud de prueba: {result.metrics['accuracy'] * 100:.2f}%")
    print(f"F1 macro:            {result.metrics['macro_f1'] * 100:.2f}%")
    print(f"Mejor época (val):   {result.best_epoch} · val_acc {result.metrics.get('best_val_accuracy', 0) * 100:.2f}%")
    print(f"Duración:            {elapsed:.1f}s")
    print(f"Modelo guardado en:  {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
