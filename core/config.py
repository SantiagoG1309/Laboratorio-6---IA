"""Configuración central de la aplicación AquaCNN · Laboratorio 6."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

APP_NAME = "AquaCNN"
APP_TITLE = "AquaCNN · Laboratorio 6"
APP_SUBTITLE = "Tratamiento de imágenes para la clasificación de peces"
APP_VERSION = "1.2.0"

UNIVERSITY = "Universidad de Cundinamarca"
UNIVERSITY_CAMPUS = "Sede Facatativá"
UNIVERSITY_CITY = "Facatativá · Cundinamarca · Colombia"
UNIVERSITY_FULL = f"{UNIVERSITY} · {UNIVERSITY_CAMPUS}"
APP_AUTHORS = (
    "Kevin Santiago García Ospina",
    "Julián Andrés Pérez Mendigaño",
    "Iván Daniel Palacio Hernández",
    "Andrés Felipe Maldonado Forero",
)

CLASSES = ("trucha", "mojarra", "pavon")
CLASS_LABELS = {"trucha": "Trucha", "mojarra": "Mojarra", "pavon": "Pavón"}
# La red recibe color (3 canales) y devuelve una neurona de salida por clase.
INPUT_CHANNELS = 3
CLASS_SCIENTIFIC = {
    "trucha": "Oncorhynchus mykiss",
    "mojarra": "Oreochromis niloticus",
    "pavon": "Cichla ocellaris",
}
CLASS_COLORS = {
    "trucha": "#38bdf8",
    "mojarra": "#34d399",
    "pavon": "#fbbf24",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Tope de seguridad: el laboratorio está pensado para cientos o pocos miles de
# imágenes; por encima de 10 000 el entrenamiento en CPU deja de ser razonable.
MAX_DATASET_IMAGES = 10_000
# Miniaturas que se generan por clase al escanear (el resto se lista sin icono).
THUMBS_PER_CLASS = 60

# ---------------------------------------------------------------------------
# Kernels de convolución clásicos
# ---------------------------------------------------------------------------
KERNELS: dict[str, dict[str, Any]] = {
    "vertical": {
        "label": "Borde vertical",
        "matrix": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        "description": "Resalta cambios horizontales de intensidad (bordes verticales).",
    },
    "horizontal": {
        "label": "Borde horizontal",
        "matrix": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        "description": "Resalta cambios verticales de intensidad (bordes horizontales).",
    },
    "sobel_x": {
        "label": "Sobel X",
        "matrix": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "description": "Gradiente horizontal suavizado: detección de bordes verticales.",
    },
    "sobel_y": {
        "label": "Sobel Y",
        "matrix": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "description": "Gradiente vertical suavizado: detección de bordes horizontales.",
    },
    "laplacian": {
        "label": "Laplaciano",
        "matrix": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        "description": "Segunda derivada: resalta contornos y texturas finas.",
    },
    "sharpen": {
        "label": "Enfoque (sharpen)",
        "matrix": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "description": "Aumenta el contraste local para resaltar detalles.",
    },
    "emboss": {
        "label": "Relieve (emboss)",
        "matrix": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        "description": "Genera un efecto de relieve tridimensional.",
    },
    "custom": {
        "label": "Personalizado",
        "matrix": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        "description": "Define tu propio kernel de 3×3.",
    },
}

COLOR_SPACES = {
    "rgb": "RGB",
    "gray": "Escala de grises",
    "hsv": "HSV",
}
EQUALIZATIONS = {
    "none": "Sin mejora",
    "hist": "Ecualización de histograma",
}
POOLINGS = {"max": "Max Pooling", "avg": "Average Pooling", "none": "Sin pooling"}


# ---------------------------------------------------------------------------
# Etapas del pipeline: todas se pueden activar o desactivar en caliente.
# ---------------------------------------------------------------------------
PIPELINE_STAGES: tuple[dict[str, Any], ...] = (
    {
        "key": "read",
        "label": "Adquisición",
        "short": "Imagen RGB",
        "icon": "image",
        "always": True,
        "description": "Lectura del archivo y decodificación a una matriz H×W×3 en RGB.",
    },
    {
        "key": "channels",
        "label": "Espacios de color",
        "short": "R · G · B / H · S · V",
        "icon": "layers",
        "always": False,
        "description": "Descomposición en canales R, G, B y H, S, V para análisis visual.",
    },
    {
        "key": "roi",
        "label": "Segmentación Otsu",
        "short": "Contorno mayor → ROI",
        "icon": "maximize",
        "always": False,
        "description": "Umbral de Otsu → contorno más grande → caja con margen; localiza el pez y elimina el fondo.",
    },
    {
        "key": "resize",
        "label": "Redimensionado",
        "short": "Escala uniforme",
        "icon": "grid",
        "always": False,
        "description": "Reescala la ROI a N×N conservando la proporción del pez y rellena el resto.",
    },
    {
        "key": "conversion",
        "label": "Conversión de color",
        "short": "Gris / HSV / RGB",
        "icon": "sliders",
        "always": False,
        "description": "Convierte a escala de grises, al canal V de HSV o conserva los 3 canales RGB.",
    },
    {
        "key": "enhance",
        "label": "Mejora de contraste",
        "short": "Histograma",
        "icon": "wave",
        "always": False,
        "description": "Ecualización global del histograma sobre cada canal RGB.",
    },
    {
        "key": "normalize",
        "label": "Normalización",
        "short": "[0, 1]",
        "icon": "sliders",
        "always": False,
        "description": "Divide entre 255 para que la red reciba valores en [0, 1].",
    },
    {
        "key": "conv",
        "label": "Convolución",
        "short": "Kernel 3×3",
        "icon": "grid",
        "always": False,
        "description": "Producto punto entre la ventana y el kernel, con padding y stride.",
    },
    {
        "key": "relu",
        "label": "Activación ReLU",
        "short": "max(0, x)",
        "icon": "wave",
        "always": False,
        "description": "Anula valores negativos y conserva las activaciones positivas.",
    },
    {
        "key": "pool",
        "label": "Pooling",
        "short": "Max / Average",
        "icon": "target",
        "always": False,
        "description": "Reduce la resolución espacial conservando la respuesta dominante.",
    },
    {
        "key": "flatten",
        "label": "Flatten",
        "short": "Vector 1D",
        "icon": "arrow",
        "always": False,
        "description": "Convierte el mapa 2D en un vector de características.",
    },
    {
        "key": "net",
        "label": "Entrada a la red",
        "short": "Tensor RGB de la CNN",
        "icon": "brain",
        "always": True,
        "description": "Tensor RGB (3 × N × N) normalizado que consume el clasificador neuronal.",
    },
)

STAGE_FIELDS: dict[str, str] = {
    "channels": "channels_enabled",
    "roi": "roi_enabled",
    "resize": "resize_enabled",
    "conversion": "conversion_enabled",
    "enhance": "enhance_enabled",
    "normalize": "normalize",
    "conv": "conv_enabled",
    "relu": "relu",
    "pool": "pool_enabled",
    "flatten": "flatten",
}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "sí", "si", "on")
    return default


def _as_int(value: Any, default: int, lo: int | None = None, hi: int | None = None) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        out = int(default)
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _as_float(value: Any, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if out != out or out in (float("inf"), float("-inf")):  # NaN / infinito
        out = float(default)
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _choice(value: Any, options: dict, default: str) -> str:
    return value if isinstance(value, str) and value in options else default


def _valid_kernel(kernel: Any) -> bool:
    try:
        arr = [[float(v) for v in row] for row in kernel]
    except (TypeError, ValueError):
        return False
    return len(arr) == 3 and all(len(row) == 3 for row in arr)


@dataclass
class PipelineConfig:
    """Parámetros del pipeline de tratamiento de imagen.

    Cada etapa tiene un interruptor propio (``*_enabled``) para poder saltarla
    y observar su efecto sobre las siguientes etapas.
    """

    color_space: str = "rgb"        # rgb | gray | hsv (solo análisis visual)
    channels_enabled: bool = True

    roi_enabled: bool = True
    roi_margin: float = 0.15        # margen alrededor de la caja del pez (0-0.4)
    resize_enabled: bool = True
    resize: int = 128

    conversion_enabled: bool = True
    enhance_enabled: bool = True
    normalize: bool = True

    equalization: str = "hist"

    conv_enabled: bool = True
    kernel_name: str = "vertical"
    kernel: list[list[float]] = field(default_factory=lambda: [row[:] for row in KERNELS["vertical"]["matrix"]])
    padding: int = 1
    stride: int = 1
    relu: bool = True

    pool_enabled: bool = True
    pooling: str = "max"
    pool_size: int = 2
    pool_stride: int = 2
    flatten: bool = True

    # Tamaño del tensor que consume el clasificador (se sincroniza desde
    # ``TrainingConfig.input_size`` para mostrar la entrada real de la red).
    net_input_size: int = 48

    # Alias retrocompatible con versiones anteriores de la configuración.
    show_channels: bool = True

    def copy(self) -> "PipelineConfig":
        return PipelineConfig(**asdict(self))

    def kernel_label(self) -> str:
        return KERNELS.get(self.kernel_name, KERNELS["custom"])["label"]

    def stage_enabled(self, key: str) -> bool:
        if key in ("read", "net"):
            return True
        if key == "pool":
            return bool(self.pool_enabled and self.pooling != "none")
        field_name = STAGE_FIELDS.get(key)
        return bool(getattr(self, field_name, True)) if field_name else True

    def set_stage(self, key: str, enabled: bool) -> None:
        field_name = STAGE_FIELDS.get(key)
        if field_name:
            setattr(self, field_name, bool(enabled))
        if key == "channels":
            self.show_channels = bool(enabled)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["show_channels"] = self.channels_enabled
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Construye la configuración tolerando JSON antiguo o incompleto."""
        data = data or {}
        allowed = cls.__dataclass_fields__.keys()
        clean = {k: v for k, v in data.items() if k in allowed}
        # Compatibilidad: versiones previas solo tenían ``show_channels``.
        if "channels_enabled" not in clean and "show_channels" in data:
            clean["channels_enabled"] = _as_bool(data.get("show_channels"), True)
        clean["channels_enabled"] = _as_bool(clean.get("channels_enabled"), True)
        clean["show_channels"] = clean["channels_enabled"]

        for key in (
            "roi_enabled", "resize_enabled", "conversion_enabled", "enhance_enabled",
            "normalize", "conv_enabled", "relu", "pool_enabled", "flatten",
        ):
            clean[key] = _as_bool(clean.get(key), True)

        clean["color_space"] = _choice(clean.get("color_space"), COLOR_SPACES, "rgb")
        clean["equalization"] = _choice(clean.get("equalization"), EQUALIZATIONS, "hist")
        clean["pooling"] = _choice(clean.get("pooling"), POOLINGS, "max")
        if clean["pooling"] == "none":
            clean["pool_enabled"] = False

        clean["roi_margin"] = _as_float(clean.get("roi_margin"), 0.15, 0.0, 0.40)
        clean["resize"] = _as_int(clean.get("resize"), 128, 32, 512)
        clean["padding"] = _as_int(clean.get("padding"), 1, 0, 8)
        clean["stride"] = _as_int(clean.get("stride"), 1, 1, 8)
        clean["pool_size"] = _as_int(clean.get("pool_size"), 2, 1, 8)
        clean["pool_stride"] = _as_int(clean.get("pool_stride"), 2, 1, 8)
        clean["net_input_size"] = _as_int(clean.get("net_input_size"), 48, 16, 256)
        clean["kernel_name"] = _choice(clean.get("kernel_name"), KERNELS, "vertical")
        kernel = clean.get("kernel")
        if not _valid_kernel(kernel):
            clean["kernel"] = [row[:] for row in KERNELS[clean["kernel_name"]]["matrix"]]
        return cls(**clean)


@dataclass
class TrainingConfig:
    """Hiperparámetros del clasificador neuronal convolucional (CNN)."""

    model_type: str = "cnn"  # Arquitectura fija: CNN convolucional
    input_size: int = 48
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    hidden_units: int = 64
    conv_filters: int = 6
    conv_filters2: int = 12
    dropout: float = 0.25
    label_smoothing: float = 0.05
    augmentation: bool = True
    class_weights: bool = True
    roi_crop: bool = True
    roi_margin: float = 0.15
    seed: int = 42

    equalization: str = "hist"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Normaliza hiperparámetros para que un JSON editado no rompa la app."""
        data = data or {}
        allowed = cls.__dataclass_fields__.keys()
        clean = {k: v for k, v in data.items() if k in allowed}
        clean["model_type"] = "cnn"
        clean["input_size"] = _as_int(clean.get("input_size"), 48, 16, 256)
        clean["epochs"] = _as_int(clean.get("epochs"), 30, 1, 1000)
        clean["batch_size"] = _as_int(clean.get("batch_size"), 16, 1, 512)
        clean["hidden_units"] = _as_int(clean.get("hidden_units"), 64, 4, 2048)
        clean["conv_filters"] = _as_int(clean.get("conv_filters"), 6, 1, 128)
        clean["conv_filters2"] = _as_int(clean.get("conv_filters2"), 12, 1, 256)
        clean["seed"] = _as_int(clean.get("seed"), 42, 0, 2**31 - 1)
        clean["learning_rate"] = _as_float(clean.get("learning_rate"), 0.01, 1e-6, 1.0)
        clean["momentum"] = _as_float(clean.get("momentum"), 0.9, 0.0, 0.999)
        clean["weight_decay"] = _as_float(clean.get("weight_decay"), 1e-4, 0.0, 1.0)
        clean["dropout"] = _as_float(clean.get("dropout"), 0.25, 0.0, 0.9)
        clean["label_smoothing"] = _as_float(clean.get("label_smoothing"), 0.05, 0.0, 0.5)
        clean["equalization"] = _choice(clean.get("equalization"), EQUALIZATIONS, "hist")
        clean["roi_margin"] = _as_float(clean.get("roi_margin"), 0.15, 0.0, 0.40)
        for key in ("augmentation", "class_weights", "roi_crop"):
            clean[key] = _as_bool(clean.get(key), True)
        clean["train_ratio"] = _as_float(clean.get("train_ratio"), 0.70, 0.0, 1.0)
        clean["val_ratio"] = _as_float(clean.get("val_ratio"), 0.15, 0.0, 1.0)
        clean["test_ratio"] = _as_float(clean.get("test_ratio"), 0.15, 0.0, 1.0)
        if clean["train_ratio"] + clean["val_ratio"] + clean["test_ratio"] <= 0:
            clean["train_ratio"], clean["val_ratio"], clean["test_ratio"] = 0.70, 0.15, 0.15
        return cls(**clean)
