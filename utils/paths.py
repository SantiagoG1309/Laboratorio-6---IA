"""Rutas portables para ejecución normal y para el ejecutable de PyInstaller.

La aplicación prefiere guardar los datos junto al ejecutable (modo portable),
pero si esa carpeta no es escribible (por ejemplo, ``C:\\Program Files``) usa
``%LOCALAPPDATA%\\AquaCNN`` para no fallar nunca al arrancar.
"""
from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

from core.config import CLASSES, INPUT_CHANNELS

APP_DIR_NAME = "AquaCNN"
MODEL_GLOB = "*.npz"
BUNDLED_MODEL_NAME = "AquaCNN_cnn_peces.npz"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", project_root()))


def resource_path(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)


def _is_writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".aquacnn_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def data_root() -> Path:
    """Carpeta base escribible para dataset, modelos y exportaciones."""
    if not is_frozen():
        return project_root()
    exe_dir = project_root()
    if _is_writable(exe_dir):
        return exe_dir
    fallback = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_DIR_NAME
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def dataset_dir() -> Path:
    """Carpeta del dataset del usuario."""
    return data_root() / "Dataset Imagenes"


def _bundled_dataset() -> Path:
    return resource_path("Dataset Imagenes")


def dataset_needs_copy() -> bool:
    """True si hay un dataset empaquetado pendiente de copiar la primera vez."""
    bundled = _bundled_dataset()
    target = dataset_dir()
    try:
        if not bundled.exists() or bundled.resolve() == target.resolve():
            return False
    except OSError:
        return False
    return not (target / ".dataset_inicializado").exists()


def copy_bundled_dataset(progress: Callable[[int, int, str], None] | None = None) -> int:
    """Copia imágenes y créditos del dataset incluido. Devuelve cuántas copió."""
    bundled = _bundled_dataset()
    target = dataset_dir()
    ensure_structure(target)
    copied = 0
    files: list[tuple[Path, Path]] = []
    if bundled.exists():
        for cls in CLASSES:
            src_dir = bundled / cls
            if not src_dir.is_dir():
                continue
            for f in sorted(src_dir.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    files.append((f, target / cls / f.name))
        for f in sorted(bundled.iterdir()):
            if f.is_file() and f.suffix.lower() in (".txt", ".md"):
                files.append((f, target / f.name))
    total = len(files)
    for i, (src, dest) in enumerate(files, start=1):
        try:
            if not dest.exists():
                shutil.copy2(src, dest)
                copied += 1
        except OSError:
            continue  # un archivo bloqueado no debe cancelar la copia completa
        if progress:
            progress(i, total, src.name)
    try:
        (target / ".dataset_inicializado").write_text("ok", encoding="utf-8")
    except OSError:
        pass
    return copied


def ensure_structure(target: str | Path | None = None) -> Path:
    """Crea las carpetas de las tres clases."""
    folder = Path(target) if target is not None else dataset_dir()
    for cls in CLASSES:
        (folder / cls).mkdir(parents=True, exist_ok=True)
    return folder


def ensure_dataset(progress: Callable[[int, int, str], None] | None = None) -> Path:
    """Prepara el dataset del usuario copiando el incluido solo la primera vez."""
    target = dataset_dir()
    ensure_structure(target)
    if dataset_needs_copy():
        copy_bundled_dataset(progress)
    return target


def models_dir() -> Path:
    path = data_root() / "Modelos entrenados"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_channels(path: Path) -> int | None:
    """Lee ``in_channels`` de un .npz sin cargar los pesos; None si no se puede."""
    try:
        import json

        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if "meta" not in data:
                return None
            return int(json.loads(str(data["meta"])).get("in_channels", 1))
    except Exception:  # noqa: BLE001
        return None


def ensure_models() -> Path:
    """Copia o actualiza los modelos incluidos en la carpeta del usuario."""
    target = models_dir()
    bundled = resource_path("Modelos entrenados")
    try:
        same = bundled.resolve() == target.resolve()
    except OSError:
        same = True
    if bundled.exists() and not same:
        for f in bundled.iterdir():
            if not (f.is_file() and f.suffix.lower() == ".npz"):
                continue
            dest = target / f.name
            try:
                # Actualiza si el modelo empaquetado es más nuevo o si la copia
                # local es de una versión antigua (por ejemplo, escala de grises).
                replace = not dest.exists() or f.stat().st_mtime > dest.stat().st_mtime + 1
                if not replace:
                    local_ch = _model_channels(dest)
                    replace = local_ch is not None and local_ch != INPUT_CHANNELS
                if replace:
                    shutil.copy2(f, dest)
            except OSError:
                continue
    return target


def default_model_path() -> Path | None:
    """Elige el modelo con el que arranca la app.

    Prioriza el modelo incluido (``AquaCNN_cnn_peces.npz``) para que la primera
    evaluación sea siempre reproducible; si no existe, usa el más reciente.
    """
    folder = models_dir()
    bundled = folder / BUNDLED_MODEL_NAME
    if bundled.exists():
        return bundled
    models = [p for p in folder.glob(MODEL_GLOB) if p.is_file()]
    if not models:
        return None
    return max(models, key=lambda p: p.stat().st_mtime)


def exports_dir() -> Path:
    path = data_root() / "Exportaciones"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(data_root()))) / APP_DIR_NAME / "cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def settings_dir() -> Path:
    base = Path(os.environ.get("APPDATA", str(data_root()))) / APP_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base
