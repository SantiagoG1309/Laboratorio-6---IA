"""Pipeline completo de tratamiento digital de imágenes (Laboratorio 6).

Flujo:  RGB → canales → ROI → redimensionado → conversión de color →
mejora → normalización → convolución → ReLU → pooling → flatten → red.

Cada etapa se puede activar o desactivar en caliente; cuando se desactiva,
la imagen pasa sin cambios a la siguiente etapa y así se refleja en el
informe de etapas (``PipelineResult.stages``).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core import image_ops as ops
from core.config import PIPELINE_STAGES, PipelineConfig


@dataclass
class PipelineResult:
    path: str
    config: dict[str, Any]
    original: np.ndarray
    channels: np.ndarray
    roi_rgb: np.ndarray
    work_rgb: np.ndarray
    gray: np.ndarray
    converted: np.ndarray           # salida de la conversión (2D o 3D)
    converted_display: np.ndarray   # uint8 para visualizar la conversión
    processed: np.ndarray           # float32 tras mejora/normalización
    processed_u8: np.ndarray        # uint8 para visualizar
    normalized_display: np.ndarray  # uint8 para visualizar la normalización
    conv_input: np.ndarray
    conv_raw: np.ndarray
    conv_relu: np.ndarray
    conv_display: np.ndarray
    relu_display: np.ndarray
    conv_channels: list[np.ndarray]
    pooled: np.ndarray
    pooled_display: np.ndarray
    flat: np.ndarray
    flat_image: np.ndarray
    net_input: np.ndarray           # tensor exacto que recibe la CNN
    net_display: np.ndarray
    hist_before: np.ndarray
    hist_after: np.ndarray
    roi_rect: tuple[int, int, int, int]
    stages: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def flat_size(self) -> int:
        return int(self.flat.size)

    @property
    def stage_states(self) -> dict[str, bool]:
        return {s["key"]: bool(s["enabled"]) for s in self.stages}


def _select_work_channel(rgb: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "hsv":
        return ops.to_hsv(rgb)[:, :, 2]
    return ops.to_gray(rgb)


_ORIGINAL_CACHE: dict[str, tuple[int, np.ndarray]] = {}
_CHANNEL_CACHE: dict[str, np.ndarray] = {}
_CACHE_LOCK = threading.Lock()
# Las imágenes originales se guardan a resolución completa, así que la caché se
# limita por bytes (y por número de entradas) para no crecer sin control al
# navegar por un dataset grande.
_CACHE_BUDGET_BYTES = 96 * 1024 * 1024
_CACHE_MAX_ENTRIES = 8


def _trim_caches(extra_bytes: int = 0) -> None:
    """Descarta las entradas más antiguas hasta respetar el presupuesto."""
    def total_bytes() -> int:
        return sum(arr.nbytes for _stamp, arr in _ORIGINAL_CACHE.values())

    while _ORIGINAL_CACHE and (
        len(_ORIGINAL_CACHE) >= _CACHE_MAX_ENTRIES
        or total_bytes() + extra_bytes > _CACHE_BUDGET_BYTES
    ):
        victim = next(iter(_ORIGINAL_CACHE))
        _ORIGINAL_CACHE.pop(victim, None)
        _CHANNEL_CACHE.pop(victim, None)


def _cached_original(path: Path) -> np.ndarray:
    key = str(path.resolve())
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    with _CACHE_LOCK:
        hit = _ORIGINAL_CACHE.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    img = ops.imread_rgb(path)
    with _CACHE_LOCK:
        _CHANNEL_CACHE.pop(key, None)
        _trim_caches(img.nbytes)
        _ORIGINAL_CACHE[key] = (stamp, img)
    return img


def _cached_channels(path: Path, rgb: np.ndarray) -> np.ndarray:
    key = str(path.resolve())
    with _CACHE_LOCK:
        hit = _CHANNEL_CACHE.get(key)
        if hit is not None:
            return hit
    comp = ops.channels_composite(rgb, size=112)
    with _CACHE_LOCK:
        _CHANNEL_CACHE[key] = comp
    return comp


def _stage_label(key: str) -> dict[str, Any]:
    return next((s for s in PIPELINE_STAGES if s["key"] == key), {"label": key, "short": "", "description": ""})


def _shape_text(arr: np.ndarray | None) -> str:
    if arr is None:
        return "—"
    if arr.ndim == 1:
        return f"vector de {arr.size}"
    if arr.ndim == 2:
        return f"{arr.shape[1]} × {arr.shape[0]}"
    return f"{arr.shape[1]} × {arr.shape[0]} × {arr.shape[2]}"


def _cap_work(img: np.ndarray, max_side: int = 1024) -> tuple[np.ndarray, bool]:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, False
    scale = max_side / longest
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA), True


def analyze_image(path: str | Path, cfg: PipelineConfig) -> PipelineResult:
    """Ejecuta el pipeline completo sobre una imagen, etapa por etapa."""
    t0 = time.perf_counter()
    path = Path(path)
    t_read = time.perf_counter()
    original = _cached_original(path)
    read_ms = (time.perf_counter() - t_read) * 1000
    timings: dict[str, float] = {}
    stages: list[dict[str, Any]] = []

    def add_stage(key: str, enabled: bool, arr: np.ndarray | None, detail: str, ms: float = 0.0) -> None:
        meta = _stage_label(key)
        stages.append({
            "key": key,
            "label": meta.get("label", key),
            "short": meta.get("short", ""),
            "description": meta.get("description", ""),
            "enabled": bool(enabled),
            "shape": _shape_text(arr),
            "detail": detail,
            "ms": round(float(ms), 2),
        })

    # 1. Adquisición -----------------------------------------------------
    add_stage("read", True, original, f"{original.shape[1]}×{original.shape[0]} px · uint8 RGB", read_ms)

    # 2. Canales RGB/HSV ------------------------------------------------
    t = time.perf_counter()
    if cfg.channels_enabled:
        channels = _cached_channels(path, original)
        detail = "R · G · B | H · S · V (cuadrícula 2×3)"
    else:
        channels = original
        detail = "Etapa desactivada: se conserva la imagen original"
    timings["canales"] = (time.perf_counter() - t) * 1000
    add_stage("channels", cfg.channels_enabled, channels, detail, timings["canales"])

    # 3. ROI ------------------------------------------------------------
    t = time.perf_counter()
    if cfg.roi_enabled:
        roi_rgb, roi_rect = ops.crop_roi(original, cfg.roi_zoom, cfg.roi_x, cfg.roi_y)
        roi_detail = f"ROI {roi_rect[2]}×{roi_rect[3]} px en ({roi_rect[0]}, {roi_rect[1]})"
    else:
        roi_rgb, roi_rect = original, (0, 0, original.shape[1], original.shape[0])
        roi_detail = "Etapa desactivada: imagen completa"
    timings["roi"] = (time.perf_counter() - t) * 1000
    add_stage("roi", cfg.roi_enabled, roi_rgb, roi_detail, timings["roi"])

    # 4. Redimensionado -------------------------------------------------
    t = time.perf_counter()
    capped = False
    if cfg.resize_enabled:
        work_rgb = ops.resize(roi_rgb, cfg.resize)
        resize_detail = f"{roi_rgb.shape[1]}×{roi_rgb.shape[0]} → {cfg.resize}×{cfg.resize}"
    else:
        work_rgb, capped = _cap_work(roi_rgb)
        resize_detail = "Etapa desactivada: se conserva la resolución de la ROI"
        if capped:
            resize_detail += f" (limitada a {work_rgb.shape[1]}×{work_rgb.shape[0]} para mantener la interfaz fluida)"
    timings["resize"] = (time.perf_counter() - t) * 1000
    add_stage("resize", cfg.resize_enabled, work_rgb, resize_detail, timings["resize"])

    # 5. Conversión de color -------------------------------------------
    t = time.perf_counter()
    if cfg.conversion_enabled and cfg.color_space == "hsv":
        converted = ops.to_hsv(work_rgb)[:, :, 2]
        conv_detail = "HSV → canal V (valor / brillo)"
    elif cfg.conversion_enabled and cfg.color_space == "gray":
        converted = ops.to_gray(work_rgb)
        conv_detail = "RGB → escala de grises (luminancia BT.601)"
    else:
        converted = work_rgb
        conv_detail = "Etapa desactivada: se conservan los 3 canales RGB"
    timings["conversion"] = (time.perf_counter() - t) * 1000
    add_stage("conversion", cfg.conversion_enabled, converted, conv_detail, timings["conversion"])
    converted_display = converted if converted.ndim == 3 else converted.astype(np.uint8)

    # 6. Mejora de contraste -------------------------------------------
    t = time.perf_counter()
    gray_full = ops.to_gray(converted)
    hist_before = ops.histogram(gray_full)
    if cfg.enhance_enabled:
        if converted.ndim == 2:
            processed_u8 = ops.equalize(converted, cfg.equalization, cfg.clahe_clip)
        else:
            processed_u8 = np.stack(
                [ops.equalize(converted[:, :, c], cfg.equalization, cfg.clahe_clip) for c in range(3)],
                axis=2,
            )
        enhance_detail = f"{cfg.equalization.upper()} · clip {cfg.clahe_clip:g}"
    else:
        processed_u8 = np.clip(converted, 0, 255).astype(np.uint8)
        enhance_detail = "Etapa desactivada: sin ecualización"
    hist_after = ops.histogram(ops.to_gray(processed_u8))
    timings["mejora"] = (time.perf_counter() - t) * 1000
    add_stage("enhance", cfg.enhance_enabled, processed_u8, enhance_detail, timings["mejora"])

    # 7. Normalización --------------------------------------------------
    t = time.perf_counter()
    if cfg.normalize:
        processed = processed_u8.astype(np.float32) / 255.0
        norm_detail = "uint8 [0, 255] → float32 [0, 1]"
    else:
        processed = processed_u8.astype(np.float32)
        norm_detail = "Etapa desactivada: se conserva el rango [0, 255]"
    timings["normalize"] = (time.perf_counter() - t) * 1000
    add_stage("normalize", cfg.normalize, processed, norm_detail, timings["normalize"])
    scale = 255.0 if float(np.nanmax(processed, initial=0.0)) <= 1.5 else 1.0
    normalized_display = np.clip(processed * scale, 0, 255).astype(np.uint8)

    # 8. Convolución ----------------------------------------------------
    t = time.perf_counter()
    conv_channels: list[np.ndarray] = []
    if cfg.conv_enabled:
        if processed.ndim == 2:
            conv_channels = [ops.convolve2d(processed, cfg.kernel, cfg.padding, cfg.stride)]
            conv_input = processed
        else:
            conv_channels = [
                ops.convolve2d(processed[:, :, c], cfg.kernel, cfg.padding, cfg.stride)
                for c in range(processed.shape[2])
            ]
            conv_input = processed
        conv_raw = np.mean(np.stack(conv_channels, axis=0), axis=0) if len(conv_channels) > 1 else conv_channels[0]
        conv_detail = (
            f"{cfg.kernel_label()} · padding {cfg.padding} · stride {cfg.stride} · "
            f"{len(conv_channels)} mapa(s)"
        )
    else:
        conv_input = processed
        conv_raw = ops.to_gray(processed) if processed.ndim == 3 else processed
        conv_channels = [conv_raw]
        conv_detail = "Etapa desactivada: se conserva la imagen mejorada"
    timings["convolucion"] = (time.perf_counter() - t) * 1000
    add_stage("conv", cfg.conv_enabled, conv_raw, conv_detail, timings["convolucion"])

    # 9. ReLU -----------------------------------------------------------
    t = time.perf_counter()
    if cfg.relu:
        conv_relu = ops.relu(conv_raw)
        relu_detail = "max(0, x): conserva solo activaciones positivas"
    else:
        conv_relu = conv_raw
        relu_detail = "Etapa desactivada: se conservan todos los valores"
    conv_display = ops.apply_colormap(conv_raw, "turbo")
    relu_display = ops.apply_colormap(conv_relu, "viridis")
    timings["relu"] = (time.perf_counter() - t) * 1000
    add_stage("relu", bool(cfg.relu), conv_relu, relu_detail, timings["relu"])

    # 10. Pooling -------------------------------------------------------
    t = time.perf_counter()
    if cfg.pool_enabled and cfg.pooling != "none":
        pooled = ops.pool2d(conv_relu, cfg.pooling, cfg.pool_size, cfg.pool_stride)
        pool_detail = f"{cfg.pooling.upper()} {cfg.pool_size}×{cfg.pool_size} · stride {cfg.pool_stride}"
    else:
        pooled = conv_relu
        pool_detail = "Etapa desactivada: se conserva el mapa completo sin reducción"
    pooled_display = ops.apply_colormap(pooled, "turbo")
    timings["pooling"] = (time.perf_counter() - t) * 1000
    add_stage("pool", bool(cfg.pool_enabled and cfg.pooling != "none"), pooled, pool_detail, timings["pooling"])

    # 11. Flatten -------------------------------------------------------
    t = time.perf_counter()
    if cfg.flatten:
        flat = ops.flatten_map(pooled)
        flat_detail = f"{_shape_text(pooled)} → {flat.size} valores"
        flat_image = ops.vector_image(flat, 640, 120, "turbo")
    else:
        flat = np.array([], np.float32)
        flat_detail = "Etapa desactivada: mapa 2D sin aplanar"
        flat_image = ops.apply_colormap(pooled, "turbo")
    timings["flatten"] = (time.perf_counter() - t) * 1000
    add_stage("flatten", bool(cfg.flatten), flat if flat.size else pooled, flat_detail, timings["flatten"])

    # 12. Entrada a la red ---------------------------------------------
    # Se construye exactamente como en el entrenamiento: resize → gris →
    # normalización [0, 1]; así la tarjeta muestra lo que realmente ve la CNN.
    t = time.perf_counter()
    net_size = max(16, int(cfg.net_input_size))
    net_u8 = ops.to_gray(processed_u8)
    net_input = ops.normalize01(ops.resize(net_u8, net_size)).astype(np.float32)
    net_display = ops.apply_colormap(net_input, "viridis")
    timings["net"] = (time.perf_counter() - t) * 1000
    add_stage("net", True, net_input, f"Tensor {net_size}×{net_size} · float32 [0, 1]", timings["net"])

    timings["total"] = (time.perf_counter() - t0) * 1000
    return PipelineResult(
        path=str(path),
        config=cfg.to_dict(),
        original=original,
        channels=channels,
        roi_rgb=roi_rgb,
        work_rgb=work_rgb,
        gray=gray_full,
        converted=converted,
        converted_display=converted_display,
        processed=processed,
        processed_u8=processed_u8,
        normalized_display=normalized_display,
        conv_input=conv_input,
        conv_raw=conv_raw,
        conv_relu=conv_relu,
        conv_display=conv_display,
        relu_display=relu_display,
        conv_channels=conv_channels,
        pooled=pooled,
        pooled_display=pooled_display,
        flat=flat,
        flat_image=flat_image,
        net_input=net_input,
        net_display=net_display,
        hist_before=hist_before,
        hist_after=hist_after,
        roi_rect=roi_rect,
        stages=stages,
        timings=timings,
    )


def preprocess_for_training(
    path: str | Path,
    size: int,
    equalization: str = "clahe",
    clahe_clip: float = 2.0,
    roi: bool = False,
    roi_zoom: float = 0.92,
    roi_x: float = 0.5,
    roi_y: float = 0.5,
) -> np.ndarray:
    """Carga y preprocesa una imagen para la red neuronal.

    Misma secuencia que el pipeline visible: ROI → resize → gris → mejora →
    normalización. Devuelve un arreglo float32 (size, size) en [0, 1].
    """
    rgb = ops.imread_rgb(path)
    if roi:
        rgb, _ = ops.crop_roi(rgb, roi_zoom, roi_x, roi_y)
    rgb = ops.resize(rgb, size)
    gray = ops.to_gray(rgb)
    gray = ops.equalize(gray, equalization, clahe_clip)
    return ops.normalize01(gray).astype(np.float32)


def overlay_roi(rgb: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Dibuja el rectángulo de ROI sobre una copia de la imagen."""
    out = rgb.copy()
    x, y, w, h = rect
    cv2.rectangle(out, (x, y), (x + w, y + h), (34, 211, 238), 2, cv2.LINE_AA)
    return out
