"""Operaciones de visión por computador y utilidades de imagen.

Todas las funciones trabajan con arreglos RGB (uint8) salvo que se indique
lo contrario, y son seguras para rutas con caracteres no ASCII en Windows.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Lectura / escritura
# ---------------------------------------------------------------------------
def imread_rgb(path: str | Path, reduce: int = 0) -> np.ndarray:
    """Lee una imagen y la devuelve en RGB. Lanza ValueError si falla.

    ``reduce=2`` o ``4`` decodifica a la mitad/cuarto de resolución, mucho más
    rápido para miniaturas (el resultado sigue siendo suficiente).
    """
    path = Path(path)
    flags = cv2.IMREAD_COLOR
    if reduce == 2:
        flags = cv2.IMREAD_REDUCED_COLOR_2
    elif reduce == 4:
        flags = cv2.IMREAD_REDUCED_COLOR_4
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"No se pudo leer la imagen: {path.name}") from exc
    if img is None:
        raise ValueError(f"Formato de imagen no soportado: {path.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _prepare_for_save(arr: np.ndarray) -> np.ndarray:
    """Convierte a uint8 sin oscurecer imágenes float normalizadas [0, 1]."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.5:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def imwrite_rgb(path: str | Path, rgb: np.ndarray) -> bool:
    """Guarda una imagen RGB respetando rutas unicode. Devuelve True si se guardó."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower() or ".png"
        arr = _prepare_for_save(rgb)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(ext, arr)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except OSError:
        return False


def imwrite_gray(path: str | Path, gray: np.ndarray) -> bool:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(path.suffix.lower() or ".png", _prepare_for_save(gray))
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Espacios de color
# ---------------------------------------------------------------------------
def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def split_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb.ndim == 2:
        return rgb, rgb, rgb
    return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


def split_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = to_hsv(rgb)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def channel_to_rgb(channel: np.ndarray) -> np.ndarray:
    if channel.ndim == 2:
        return cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)
    return channel


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def segment_fish_roi(rgb: np.ndarray, margin: float = 0.15) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Localiza el pez con el umbral de Otsu y devuelve su caja con margen.

    Pasos: gris → desenfoque → umbral de Otsu → limpieza morfológica →
    contorno más grande → caja envolvente. Se prueban las dos polaridades del
    umbral y se elige la que produce un objeto más grande sin pegarse a todos
    los bordes. Si no hay un contorno fiable, se conserva la imagen completa.
    """
    arr = np.asarray(rgb)
    if arr.size == 0:
        return arr, (0, 0, 0, 0)
    h, w = arr.shape[:2]
    gray = to_gray(arr)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def largest(mask: np.ndarray):
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    # Otsu puede dejar el pez en blanco o en negro según la foto, así que se
    # prueban las dos polaridades. El fondo suele ser el contorno que ocupa
    # casi todo el marco: se descarta por área y por contacto con los bordes.
    best = None
    best_score = -1.0
    total = float(max(1, w * h))
    for mask in (binary, cv2.bitwise_not(binary)):
        contour = largest(mask)
        if contour is None:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 4 or ch < 4:
            continue
        area_frac = cv2.contourArea(contour) / total
        frac = (cw * ch) / total
        if area_frac < 0.03 or frac < 0.05:
            continue  # ruido o mota, no un pez
        border = int(x <= 2) + int(y <= 2) + int(x + cw >= w - 2) + int(y + ch >= h - 2)
        solidity = area_frac / max(frac, 1e-6)
        # Un pez llena su caja de forma razonable y no toca todos los bordes.
        score = solidity - 0.15 * border - 0.60 * max(0.0, frac - 0.85)
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return arr, (0, 0, w, h)

    x, y, cw, ch = best
    mx = int(round(cw * float(np.clip(margin, 0.0, 0.4))))
    my = int(round(ch * float(np.clip(margin, 0.0, 0.4))))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w, x + cw + mx)
    y1 = min(h, y + ch + my)
    return arr[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    side = max(8, int(size))
    if img.size == 0:
        return np.zeros((side, side) + tuple(img.shape[2:]), np.uint8)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > side else cv2.INTER_CUBIC
    return cv2.resize(img, (side, side), interpolation=interp)


def resize_with_pad(img: np.ndarray, size: int, pad: int | float = 0) -> np.ndarray:
    """Redimensiona conservando la proporción y rellena hasta un cuadrado.

    El pez nunca se deforma: se escala por el lado que más crece y el espacio
    restante se completa con ``pad`` (negro por defecto), centrando la imagen.
    """
    side = max(8, int(size))
    arr = np.asarray(img)
    if arr.size == 0:
        return np.full((side, side), pad, dtype=np.uint8)
    h, w = arr.shape[:2]
    scale = min(side / float(w), side / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(arr, (nw, nh), interpolation=interp)
    if arr.ndim == 2:
        canvas = np.full((side, side), pad, dtype=arr.dtype)
    else:
        canvas = np.full((side, side, arr.shape[2]), pad, dtype=arr.dtype)
    x = (side - nw) // 2
    y = (side - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def normalize01(gray: np.ndarray) -> np.ndarray:
    """Normaliza a [0, 1] según el dtype (uint8 siempre se divide entre 255)."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return arr
    if np.issubdtype(np.asarray(gray).dtype, np.integer):
        arr = arr / 255.0
    elif float(arr.max(initial=0.0)) > 1.5 or float(arr.min(initial=0.0)) < -1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), np.uint8)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-9:
        # Un mapa constante no debe verse completamente negro.
        return np.full(arr.shape, 127, np.uint8)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return np.nan_to_num(scaled, nan=0.0).clip(0, 255).astype(np.uint8)


def apply_colormap(gray: np.ndarray, name: str = "viridis") -> np.ndarray:
    """Devuelve una imagen RGB coloreada a partir de un mapa 2D."""
    g = to_gray(gray) if getattr(gray, "ndim", 0) == 3 else gray
    arr = to_uint8(g)
    code = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
        "magma": cv2.COLORMAP_MAGMA,
        "ocean": cv2.COLORMAP_OCEAN,
        "jet": cv2.COLORMAP_JET,
    }.get(name, cv2.COLORMAP_VIRIDIS)
    bgr = cv2.applyColorMap(arr, code)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Análisis y mejora
# ---------------------------------------------------------------------------
def histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    arr = np.asarray(gray)
    if arr.size and np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    hist = cv2.calcHist([np.clip(arr, 0, 255).astype(np.uint8)], [0], None, [bins], [0, 256])
    return hist.flatten()


def equalize(
    gray: np.ndarray,
    mode: str = "hist",
    clip: float = 2.0,
    tiles: int = 8,
) -> np.ndarray:
    """Ecualización global o CLAHE (contraste local) sobre una imagen 2D."""
    arr = np.asarray(gray)
    if arr.size and np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    g = np.clip(arr, 0, 255).astype(np.uint8)
    if mode == "hist":
        return cv2.equalizeHist(g)
    if mode == "clahe":
        grid = max(2, min(16, int(tiles)))
        clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip, 0.5, 8.0)), tileGridSize=(grid, grid))
        return clahe.apply(g)
    return g


def equalize_rgb(
    rgb: np.ndarray,
    mode: str = "hist",
    clip: float = 2.0,
    tiles: int = 8,
) -> np.ndarray:
    """Mejora el contraste de una imagen RGB.

    ``hist`` ecualiza cada canal por separado; ``clahe`` aplica el contraste
    adaptativo al canal L de CIELAB para no alterar el tono de los colores.
    """
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return equalize(arr, mode, clip, tiles)
    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    if mode == "clahe":
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = equalize(lab[:, :, 0], "clahe", clip, tiles)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if mode != "hist":
        return u8
    return np.stack([equalize(u8[:, :, c], "hist") for c in range(u8.shape[2])], axis=2)


# ---------------------------------------------------------------------------
# Convolución y pooling (visualización / extracción de características)
# ---------------------------------------------------------------------------
def convolve2d(
    gray: np.ndarray,
    kernel: np.ndarray | list[list[float]],
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Convolución 2D con padding y stride configurables.

    Implementada con ``sliding_window_view`` para evitar bucles por píxel.
    """
    x = np.asarray(gray, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    if k.shape != (3, 3):
        k = cv2.resize(k, (3, 3), interpolation=cv2.INTER_NEAREST)
    p = int(max(0, padding))
    s = int(max(1, stride))
    if p > 0:
        x = np.pad(x, ((p, p), (p, p)), mode="constant", constant_values=0.0)
    h, w = x.shape
    kh, kw = k.shape
    if h < kh or w < kw:
        return np.zeros((1, 1), np.float32)
    hout = (h - kh) // s + 1
    wout = (w - kw) // s + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw))[::s, ::s]
    windows = windows[:hout, :wout]
    return np.einsum("ijkl,kl->ij", windows, k, optimize=True).astype(np.float32)


def relu(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0)


def pool2d(arr: np.ndarray, mode: str = "max", size: int = 2, stride: int = 2) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if mode == "none" or x.ndim != 2:
        return x
    k = max(1, int(size))
    s = max(1, int(stride))
    h, w = x.shape
    if h < k or w < k:
        return x
    hout = (h - k) // s + 1
    wout = (w - k) // s + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (k, k))[::s, ::s][:hout, :wout]
    if mode == "avg":
        return windows.mean(axis=(-1, -2)).astype(np.float32)
    return windows.max(axis=(-1, -2)).astype(np.float32)


def flatten_map(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).ravel()


def vector_image(vector: np.ndarray, width: int = 640, height: int = 120, colormap: str = "turbo") -> np.ndarray:
    """Convierte un vector 1D en una tira coloreada legible."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    if v.size == 0:
        return np.zeros((height, width, 3), np.uint8)
    # Rejilla proporcional para que se aprecien patrones en vectores largos.
    if v.size > width * height:
        step = int(np.ceil(v.size / (width * height)))
        v = v[::step][: width * height]
    n = v.size
    cols = int(np.ceil(np.sqrt(n * width / max(1, height))))
    rows = int(np.ceil(n / cols))
    canvas = np.zeros((rows, cols), np.float32)
    canvas.flat[:n] = v
    img = to_uint8(canvas)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    code = {"viridis": cv2.COLORMAP_VIRIDIS, "turbo": cv2.COLORMAP_TURBO}.get(colormap, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(cv2.applyColorMap(img, code), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Composición de canales para visualización
# ---------------------------------------------------------------------------
def channels_composite(rgb: np.ndarray, size: int = 128) -> np.ndarray:
    """Construye una cuadrícula 2×3 con R, G, B y H, S, V etiquetados."""
    r, g, b = split_rgb(rgb)
    h, s, v = split_hsv(rgb)
    items = [
        ("R", r, (255, 99, 132)),
        ("G", g, (52, 211, 153)),
        ("B", b, (56, 189, 248)),
        ("H", h, (251, 191, 36)),
        ("S", s, (167, 139, 250)),
        ("V", v, (244, 114, 182)),
    ]
    tile = int(size)
    label_h = 24
    gap = 8
    out = np.full((2 * (tile + label_h) + gap * 3, 3 * tile + gap * 4, 3), 10, np.uint8)
    for i, (name, ch, color) in enumerate(items):
        row, col = divmod(i, 3)
        y = gap + row * (tile + label_h + gap)
        x = gap + col * (tile + gap)
        ch_rgb = cv2.cvtColor(cv2.resize(ch, (tile, tile), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
        out[y:y + tile, x:x + tile] = ch_rgb
        # etiqueta
        cv2.rectangle(out, (x, y + tile), (x + tile, y + tile + label_h), (17, 28, 45), -1)
        cv2.putText(out, name, (x + 8, y + tile + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 245, 255), 1, cv2.LINE_AA)
        cv2.circle(out, (x + tile - 14, y + tile + 12), 4, color, -1, cv2.LINE_AA)
    return out


def resize_letterbox(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Redimensiona conservando aspecto para previsualizaciones."""
    h, w = img.shape[:2]
    if h == 0 or w == 0 or max_w <= 0 or max_h <= 0:
        return img
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if (nw, nh) == (w, h):
        return img
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
