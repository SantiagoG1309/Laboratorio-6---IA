"""Entrenamiento, evaluación y predicción de los clasificadores neuronales."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import math

import cv2
import numpy as np

from core.cnn import build_model, make_optimizer
from core.config import CLASS_LABELS, CLASSES, INPUT_CHANNELS, TrainingConfig
from core.dataset import Sample
from core.metrics import evaluate_predictions
from core.pipeline import preprocess_for_training


@dataclass
class TrainingResult:
    model: object
    config: TrainingConfig
    history: dict[str, list[float]]
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    test_paths: list[str]
    test_labels: list[int]
    best_epoch: int = 0
    duration: float = 0.0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    stopped: bool = False
    extra: dict = field(default_factory=dict)


def load_split_arrays(
    samples: list[Sample],
    cfg: TrainingConfig,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Sample]]:
    """Preprocesa las imágenes; las corruptas se omiten sin abortar el lote.

    Devuelve ``(X, y, muestras_válidas)`` para que rutas y etiquetas sigan
    alineadas aunque alguna imagen no se pueda leer.
    """
    X = np.zeros(
        (len(samples), INPUT_CHANNELS, cfg.input_size, cfg.input_size), dtype=np.float32
    )
    y = np.zeros(len(samples), dtype=np.int64)
    kept: list[Sample] = []
    skipped = 0
    for i, s in enumerate(samples, start=1):
        try:
            img = preprocess_for_training(
                s.path,
                size=cfg.input_size,
                equalization=cfg.equalization,
                roi=cfg.roi_crop,
                roi_margin=cfg.roi_margin,
            )
            X[len(kept)] = img
            y[len(kept)] = s.label
            kept.append(s)
        except (ValueError, OSError, cv2.error):
            skipped += 1
        if log and i % 25 == 0:
            log(f"  preprocesadas {i}/{len(samples)} imágenes…")
    if skipped:
        log(f"  aviso: {skipped} imagen(es) no se pudieron leer y se omitieron")
    return X[:len(kept)], y[:len(kept)], kept


def class_weight_vector(y: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return weights


def _augment_batch(Xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aumenta el lote con volteos, rotaciones, zooms, brillo, ruido y cutout.

    Todo se aplica sobre imágenes normalizadas [0, 1] y en el mismo tensor, de
    modo que el coste por lote sigue siendo despreciable frente a la convolución.
    """
    import cv2

    out = Xb.copy()
    n = out.shape[0]
    if out.ndim != 4:
        return out
    for i in range(n):
        img = out[i]
        h, w = img.shape[1:]
        if rng.random() < 0.5:
            img = img[:, :, ::-1]
        if rng.random() < 0.5:
            angle = float(rng.uniform(-12.0, 12.0))
            scale = float(rng.uniform(0.92, 1.08))
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            # OpenCV espera H×W×C: se llevan los canales al final y se devuelven.
            hwc = np.transpose(img, (1, 2, 0))
            hwc = cv2.warpAffine(
                hwc, matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            img = np.transpose(hwc, (2, 0, 1))
        if rng.random() < 0.35:
            side = int(round(min(h, w) * float(rng.uniform(0.80, 1.0))))
            side = max(8, side)
            x0 = int(rng.integers(0, max(1, w - side + 1)))
            y0 = int(rng.integers(0, max(1, h - side + 1)))
            patch = np.transpose(img[:, y0:y0 + side, x0:x0 + side], (1, 2, 0))
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
            img = np.transpose(patch, (2, 0, 1))
        if rng.random() < 0.5:
            img = img * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.06, 0.06))
        if rng.random() < 0.3:
            img = img + rng.normal(0.0, 0.02, img.shape).astype(np.float32)
        if rng.random() < 0.25:
            cw = max(4, int(w * 0.22))
            ch = max(4, int(h * 0.22))
            x0 = int(rng.integers(0, max(1, w - cw + 1)))
            y0 = int(rng.integers(0, max(1, h - ch + 1)))
            img = img.copy()
            img[:, y0:y0 + ch, x0:x0 + cw] = 0.0
        out[i] = np.clip(img, 0.0, 1.0)
    return out


def _evaluate(model, X: np.ndarray, y: np.ndarray, batch: int = 32) -> tuple[float, float, np.ndarray]:
    if len(y) == 0:
        return 0.0, 0.0, np.zeros((0, 3))
    probs = []
    for start in range(0, len(X), batch):
        probs.append(model.predict_proba(X[start:start + batch]))
    p = np.vstack(probs)
    pred = p.argmax(axis=1)
    acc = float((pred == y).mean())
    loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    return loss, acc, p


def train_model(
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
    cfg: TrainingConfig,
    on_epoch: Callable[[dict], None] | None = None,
    on_batch: Callable[[int, int, int, float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingResult:
    t0 = time.perf_counter()
    log = on_log or (lambda _s: None)
    log(f"Preparando {len(train) + len(val) + len(test)} imágenes a {cfg.input_size}×{cfg.input_size}…")
    Xtr, ytr, train_ok = load_split_arrays(train, cfg, log)
    Xva, yva, val_ok = load_split_arrays(val, cfg, log)
    Xte, yte, test_ok = load_split_arrays(test, cfg, log)
    log(f"Datos listos: train={len(ytr)} · val={len(yva)} · test={len(yte)}")
    if len(ytr) < 2:
        raise ValueError("No hay suficientes imágenes de entrenamiento legibles para entrenar.")

    rng = np.random.default_rng(cfg.seed)
    model = build_model(
        cfg.model_type, cfg.input_size, cfg.hidden_units,
        cfg.conv_filters, cfg.conv_filters2, n_out=len(CLASSES), seed=cfg.seed,
        dropout=cfg.dropout,
    )
    opt = make_optimizer(model, cfg.learning_rate, cfg.momentum, cfg.weight_decay)
    weights = class_weight_vector(ytr, len(CLASSES)) if cfg.class_weights else None
    if weights is not None:
        log("Pesos de clase: " + ", ".join(f"{CLASS_LABELS[c]}={weights[i]:.2f}" for i, c in enumerate(CLASSES)))

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}
    best_acc = -1.0
    best_state: list[np.ndarray] | None = None
    best_epoch = 0
    stopped = False
    n = len(ytr)
    batch = max(1, int(cfg.batch_size))
    steps = max(1, int(np.ceil(n / batch)))

    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            stopped = True
            log("Entrenamiento detenido por el usuario.")
            break
        perm = rng.permutation(n)
        epoch_losses = []
        for step in range(steps):
            sl = slice(step * batch, min(n, (step + 1) * batch))
            idx = perm[sl]
            Xb, yb = Xtr[idx], ytr[idx]
            if cfg.augmentation and len(yb) > 1:
                Xb = _augment_batch(Xb, rng)
            sw = weights[yb] if weights is not None else None
            loss, grads = model.loss_and_grads(Xb, yb, sw, label_smoothing=cfg.label_smoothing)
            opt.step(model.params(), grads)
            epoch_losses.append(loss)
            if on_batch:
                on_batch(epoch, step + 1, steps, float(loss))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        # Evaluar todo el train cada época consume ~35 % del tiempo. Para
        # datasets grandes se usa una muestra fija representativa.
        if n <= 768:
            acc_idx = np.arange(n)
        else:
            acc_idx = rng.choice(n, 768, replace=False)
        _, train_acc, _ = _evaluate(model, Xtr[acc_idx], ytr[acc_idx])
        val_loss, val_acc, _ = _evaluate(model, Xva, yva) if len(yva) else (0.0, 0.0, np.zeros((0, 3)))
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(opt.lr)
        # Decaimiento coseno: pasos grandes al principio y finos al final, lo
        # que reduce la oscilación de la exactitud de validación.
        if cfg.epochs > 1:
            frac = epoch / cfg.epochs
            opt.lr = float(cfg.learning_rate * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * frac))))

        if len(yva) and val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = [p.copy() for _, p in model.params()]

        elapsed = time.perf_counter() - t0
        eta = elapsed / epoch * (cfg.epochs - epoch)
        info = {
            "epoch": epoch,
            "epochs": cfg.epochs,
            "loss": train_loss,
            "accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "elapsed": elapsed,
            "eta": eta,
            "lr": opt.lr,
            "best_epoch": best_epoch,
            "best_val_accuracy": max(best_acc, 0.0),
        }
        log(
            f"Época {epoch:02d}/{cfg.epochs:02d} · loss {train_loss:.4f} · acc {train_acc * 100:5.1f}%"
            f" · val_loss {val_loss:.4f} · val_acc {val_acc * 100:5.1f}%"
        )
        if on_epoch:
            on_epoch(info)

    if best_state is not None:
        for (_, p), saved in zip(model.params(), best_state):
            p[...] = saved

    # Gradientes finales sobre un lote de prueba para el inspector de pesos.
    if len(yte):
        sample = min(len(yte), 16)
        try:
            model.loss_and_grads(Xte[:sample], yte[:sample])
        except Exception:
            pass

    if len(yte):
        test_loss, test_acc, probs = _evaluate(model, Xte, yte)
        pred = probs.argmax(axis=1)
        metrics = evaluate_predictions(yte, pred, len(CLASSES))
        metrics["test_loss"] = test_loss
        metrics["best_val_accuracy"] = max(best_acc, 0.0)
        log(f"Prueba: exactitud {test_acc * 100:.2f}% · F1 macro {metrics['macro_f1'] * 100:.2f}%")
    else:
        metrics = evaluate_predictions(np.array([]), np.array([]), len(CLASSES))
        metrics["test_loss"] = 0.0
        metrics["best_val_accuracy"] = max(best_acc, 0.0)
        probs = np.zeros((0, len(CLASSES)))
        pred = np.array([], dtype=int)

    return TrainingResult(
        model=model,
        config=cfg,
        history=history,
        metrics=metrics,
        y_true=yte,
        y_pred=pred,
        probs=probs,
        test_paths=[str(s.path) for s in test_ok],
        test_labels=[s.label for s in test_ok],
        best_epoch=best_epoch,
        duration=time.perf_counter() - t0,
        train_count=len(ytr),
        val_count=len(yva),
        test_count=len(yte),
        stopped=stopped,
    )


def predict_paths(model, paths: list[str], cfg: TrainingConfig, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Predice por lotes para no reservar gigabytes con datasets grandes.

    Una imagen ilegible se sustituye por ceros (la red devuelve una
    predicción válida) en lugar de interrumpir toda la evaluación.
    """
    if not paths:
        return np.zeros((0, len(CLASSES))), np.array([], dtype=int)
    probs = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)
    step = max(1, int(batch))
    for start in range(0, len(paths), step):
        chunk = paths[start:start + step]
        X = np.zeros(
            (len(chunk), INPUT_CHANNELS, cfg.input_size, cfg.input_size), dtype=np.float32
        )
        for i, p in enumerate(chunk):
            try:
                X[i] = preprocess_for_training(
                    p, cfg.input_size, cfg.equalization,
                    roi=cfg.roi_crop, roi_margin=cfg.roi_margin,
                )
            except (ValueError, OSError, cv2.error):
                pass
        probs[start:start + len(chunk)] = model.predict_proba(X)
    return probs, probs.argmax(axis=1)
