"""Preferencias de la interfaz persistidas con ``QSettings``.

Se guardan en ``HKCU\\Software\\AquaCNN\\Laboratorio6`` en Windows y se leen
antes de construir las páginas para que el tema y el acento se apliquen desde
el primer fotograma (sin parpadeos de color).
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings

ORGANIZATION = "AquaCNN"
APPLICATION = "Laboratorio6"

DEFAULTS: dict[str, Any] = {
    "theme": "ocean",
    "accent": "cian",
    "font_family": "",
    "font_size": 13,
    "animations": True,
    "tooltips": True,
    "start_maximized": True,
    "thumbs": 60,
    "auto_pipeline": True,
}


def store() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "si", "sí", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return value


def get(key: str, default: Any | None = None) -> Any:
    fallback = DEFAULTS.get(key, default)
    raw = store().value(key, fallback)
    return _coerce(raw, fallback)


def set_value(key: str, value: Any) -> None:  # noqa: A001 - nombre corto y claro
    store().setValue(key, value)


def reset() -> None:
    s = store()
    for key, value in DEFAULTS.items():
        s.setValue(key, value)


def snapshot() -> dict[str, Any]:
    return {key: get(key, value) for key, value in DEFAULTS.items()}
