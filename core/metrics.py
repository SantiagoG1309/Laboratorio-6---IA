"""Métricas de evaluación: matriz de confusión, precisión, recall y F1."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    t = np.asarray(y_true).ravel().astype(np.int64, copy=False)
    p = np.asarray(y_pred).ravel().astype(np.int64, copy=False)
    if t.size == 0 or p.size == 0:
        return np.zeros((n_classes, n_classes), dtype=int)
    valid = (t >= 0) & (t < n_classes) & (p >= 0) & (p < n_classes)
    flat = np.bincount(t[valid] * n_classes + p[valid], minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def per_class_metrics(cm: np.ndarray) -> list[dict[str, float]]:
    n = cm.shape[0]
    out = []
    for i in range(n):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        support = int(cm[i, :].sum())
        out.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "accuracy": recall,
        })
    return out


def macro_metrics(cm: np.ndarray, rows: list[dict[str, float]] | None = None) -> dict[str, float]:
    rows = rows if rows is not None else per_class_metrics(cm)
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "f1": float(np.mean([r["f1"] for r in rows])),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    acc = accuracy(y_true, y_pred)
    rows = per_class_metrics(cm)
    macro = macro_metrics(cm, rows)
    return {
        "accuracy": acc,
        "confusion": cm,
        "per_class": rows,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "support": int(np.asarray(y_true).size),
    }


def report_lines(metrics: dict, labels: list[str]) -> list[str]:
    lines = [
        f"Exactitud global: {metrics['accuracy'] * 100:.2f}%",
        f"Precisión macro:  {metrics['macro_precision'] * 100:.2f}%",
        f"Recall macro:     {metrics['macro_recall'] * 100:.2f}%",
        f"F1 macro:         {metrics['macro_f1'] * 100:.2f}%",
        "",
        f"{'Clase':<12}{'Precisión':>10}{'Recall':>9}{'F1':>9}{'Soporte':>9}",
    ]
    for label, row in zip(labels, metrics["per_class"]):
        lines.append(
            f"{label:<12}{row['precision'] * 100:>9.1f}%{row['recall'] * 100:>8.1f}%"
            f"{row['f1'] * 100:>8.1f}%{row['support']:>9d}"
        )
    return lines
