"""Glosario breve de términos técnicos y tooltips por palabra."""
from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QLabel, QToolTip, QWidget

from ui.theme import TEXT, rgba

# Definiciones cortas: una línea, sin rodeos.
GLOSSARY: dict[str, str] = {
    "CNN": "Red convolucional: filtra la imagen para detectar patrones.",
    "Kernel": "Matriz 3×3 que se desliza sobre la imagen y detecta un patrón.",
    "Stride": "Cuántos píxeles salta la ventana al recorrer la imagen.",
    "Padding": "Borde de ceros para no perder las orillas de la imagen.",
    "Convolución": "Producto punto entre una ventana de la imagen y el kernel.",
    "ReLU": "Activación max(0, x): descarta los valores negativos.",
    "Pooling": "Reduce el mapa quedándose con un resumen de cada ventana.",
    "Max Pooling": "Toma el valor máximo de cada ventana.",
    "Average Pooling": "Toma el promedio de cada ventana.",
    "Flatten": "Aplana los mapas 2D en un vector 1D.",
    "Oculta": "Capa de backpropagation: cada neurona combina todas las activaciones anteriores.",
    "Capa oculta": "Cada neurona calcula z = x·W + b y aplica ReLU antes de pasar a la salida.",
    "Softmax": "Convierte los logits en probabilidades que suman 1.",
    "Backpropagation": "Regla de la cadena para calcular el gradiente de cada peso.",
    "Momentum": "Inercia que suaviza y acelera el descenso del gradiente.",
    "Learning rate": "Tamaño del paso al actualizar los pesos.",
    "Weight decay": "Penalización L2 que evita pesos demasiado grandes.",
    "Época": "Una pasada completa por todas las imágenes de entrenamiento.",
    "Batch": "Grupo de imágenes procesado antes de actualizar los pesos.",
    "Data augmentation": "Variaciones artificiales para que la red generalice.",
    "ROI": "Caja que encierra al pez tras la segmentación, con un margen extra.",
    "Otsu": "Umbral automático que separa el pez del fondo usando el histograma.",
    "Contorno": "Curva cerrada que rodea una región segmentada; se elige el de mayor área.",
    "HSV": "Tono, saturación y valor: separa color e iluminación.",
    "FLOPS": "Operaciones de coma flotante necesarias por imagen.",
    "Parámetros": "Pesos y sesgos que la red aprende.",
    "Overfitting": "La red memoriza el entrenamiento y falla con datos nuevos.",
    "Exactitud": "Porcentaje de predicciones correctas (accuracy).",
    "Precisión": "De lo predicho como X, cuánto era realmente X.",
    "Recall": "De las imágenes reales de X, cuántas encontró.",
    "F1": "Media armónica entre precisión y recall.",
    "Matriz de confusión": "Cruza clase real con predicción para ver confusiones.",
    "Gradiente": "Dirección en la que hay que ajustar un peso.",
    "Logits": "Salidas crudas de la última capa, antes de Softmax.",
    "Tensor": "Arreglo multidimensional de números.",
    "Actividad": "Valor de salida de una neurona para una imagen dada.",
    "Dropout": "Apaga neuronas al azar durante el entrenamiento.",
}

_TERMS_SORTED = sorted(GLOSSARY, key=len, reverse=True)
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")
_LOOKUP = {term.lower(): definition for term, definition in GLOSSARY.items()}


def tip(term: str, extra: str = "") -> str:
    """Tooltip breve de un término, con detalle opcional."""
    base = GLOSSARY.get(term, "")
    if extra:
        return (
            f"<b>{term}</b> — {base}<br>"
            f"<span style='color:{rgba(TEXT, 0.62)}'>{extra}</span>"
        )
    return f"<b>{term}</b> — {base}"


def definition(word: str) -> str | None:
    """Definición del glosario para una palabra (sin distinguir mayúsculas)."""
    return _LOOKUP.get(word.strip().lower())


def find_terms(text: str) -> list[str]:
    """Términos del glosario presentes en un texto."""
    found: list[str] = []
    for term in _TERMS_SORTED:
        if term.lower() in text.lower():
            found.append(term)
    return found


class GlossaryLabel(QLabel):
    """Etiqueta que explica el término técnico sobre el que está el cursor."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._glossary_terms = find_terms(text)

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self._glossary_terms = find_terms(text)

    # ------------------------------------------------------------------
    def _word_at(self, pos) -> str:
        text = self.text()
        if not text or "<" in text:
            return ""
        doc = QTextDocument()
        doc.setDefaultFont(self.font())
        doc.setPlainText(text)
        doc.setTextWidth(max(1, self.width() - 2))
        cursor = doc.documentLayout().hitTest(
            QPointF(pos.x(), pos.y()), Qt.HitTestAccuracy.FuzzyHit
        )
        if cursor is None or cursor < 0:
            return ""
        char = doc.characterAt(cursor)
        if not char or not _WORD_RE.match(char):
            return ""
        start = cursor
        while start > 0 and _WORD_RE.match(doc.characterAt(start - 1) or ""):
            start -= 1
        end = cursor
        length = doc.characterCount()
        while end + 1 < length and _WORD_RE.match(doc.characterAt(end + 1) or ""):
            end += 1
        return doc.toPlainText()[start : end + 1]

    def event(self, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.ToolTip and self._glossary_terms:
            word = self._word_at(event.pos())
            text = definition(word)
            if text:
                QToolTip.showText(event.globalPos(), f"<b>{word}</b> — {text}", self)
                return True
        return super().event(event)


def glossary_label(text: str = "", parent: QWidget | None = None) -> GlossaryLabel:
    """Atajo para crear una etiqueta con glosario."""
    return GlossaryLabel(text, parent)
