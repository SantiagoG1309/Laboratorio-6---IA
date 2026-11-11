"""Página de predicción libre: arrastra cualquier imagen y clasifícala.

Reutiliza exactamente el mismo preprocesado (RGB → Otsu → relleno → histograma
→ normalización) y el mismo modelo CNN que la página de Resultados, pero para
una sola imagen elegida por el usuario.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.config import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASSES,
    IMAGE_EXTENSIONS,
    TrainingConfig,
)
from core.image_ops import imread_rgb
from core.pipeline import preprocess_for_training
from ui import icons
from ui.theme import (
    ACCENT,
    BORDER,
    MUTED,
    ON_ACCENT,
    SUCCESS,
    SURFACE_2,
    TEXT,
    WARNING,
    rgba,
)
from ui.widgets.cards import Badge, Card, SectionHeader
from ui.widgets.flow_layout import ResponsiveGrid
from ui.widgets.image_view import ImageView


class DropZone(QFrame):
    """Zona de arrastre que también muestra la imagen seleccionada."""

    fileDropped = Signal(str)
    chooseRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setProperty("soft", True)
        self.setMinimumHeight(280)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        self.view = ImageView("Arrastra una imagen aquí", "upload", min_height=220)
        self.view.doubleClicked.connect(self.chooseRequested.emit)
        lay.addWidget(self.view)
        self.hint = QLabel("Arrastra y suelta una imagen, o haz clic en «Elegir imagen».")
        self.hint.setProperty("subtle", True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet("font-size: 10px;")
        lay.addWidget(self.hint)
        self._hover = False

    @staticmethod
    def _first_image(event) -> str | None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return str(path)
        return None

    def _set_hover(self, hover: bool) -> None:
        if hover == self._hover:
            return
        self._hover = hover
        color = ACCENT if hover else BORDER
        self.setStyleSheet(
            f"QFrame[soft='true'] {{ border: 2px {'dashed' if hover else 'solid'} {color};"
            f" border-radius: 12px; background: {rgba(ACCENT, 0.06) if hover else SURFACE_2}; }}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._first_image(event):
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_hover(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        path = self._first_image(event)
        self._set_hover(False)
        if path:
            event.acceptProposedAction()
            self.fileDropped.emit(path)
        else:
            event.ignore()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.chooseRequested.emit()
        super().mousePressEvent(event)


def _bar(color: str) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 1000)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedHeight(9)
    bar.setStyleSheet(
        f"QProgressBar {{ background: {SURFACE_2}; border: 1px solid {BORDER};"
        " border-radius: 4px; }}"
        f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
    )
    return bar


class PredictPage(QWidget):
    """Carga una imagen arbitraria y muestra clase, confianza y probabilidades."""

    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent)
        self.main = main
        self.path: str | None = None
        self._probs: np.ndarray | None = None
        self._bars: dict[str, QProgressBar] = {}
        self._pcts: dict[str, QLabel] = {}
        self._names: dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget()
        content.setObjectName("Page")
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 26)
        layout.setSpacing(16)

        # Cabecera -------------------------------------------------------
        head = QHBoxLayout()
        hcol = QVBoxLayout()
        title = QLabel("Predicción de imágenes")
        title.setProperty("h1", True)
        hcol.addWidget(title)
        subtitle = QLabel(
            "Arrastra una foto de un pez y la CNN dirá si es trucha, mojarra o pavón, "
            "con la confianza de cada clase."
        )
        subtitle.setProperty("muted", True)
        subtitle.setWordWrap(True)
        hcol.addWidget(subtitle)
        head.addLayout(hcol, 1)
        self.model_badge = Badge("Modelo no disponible", MUTED)
        head.addWidget(self.model_badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)

        # Acciones -------------------------------------------------------
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.choose_btn = QPushButton("Elegir imagen")
        self.choose_btn.setProperty("variant", "primary")
        self.choose_btn.setIcon(icons.icon("upload", ON_ACCENT, 16))
        self.choose_btn.clicked.connect(self.choose_image)
        self.choose_btn.setToolTip("Abre el explorador para elegir una imagen JPG, PNG, BMP, WEBP o TIFF.")
        actions.addWidget(self.choose_btn)

        self.repeat_btn = QPushButton("Predecir de nuevo")
        self.repeat_btn.setProperty("variant", "ghost")
        self.repeat_btn.setIcon(icons.icon("refresh", MUTED, 15))
        self.repeat_btn.setEnabled(False)
        self.repeat_btn.setToolTip("Vuelve a preprocesar la imagen y ejecuta la CNN otra vez.")
        self.repeat_btn.clicked.connect(self._predict)
        actions.addWidget(self.repeat_btn)

        self.clear_btn = QPushButton("Quitar")
        self.clear_btn.setProperty("variant", "ghost")
        self.clear_btn.setIcon(icons.icon("trash", MUTED, 15))
        self.clear_btn.setEnabled(False)
        self.clear_btn.setToolTip("Quita la imagen y limpia el resultado de la predicción.")
        self.clear_btn.clicked.connect(self.clear)
        actions.addWidget(self.clear_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.file_label = QLabel("Ninguna imagen seleccionada")
        self.file_label.setProperty("muted", True)
        self.file_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.file_label)

        # Tarjetas -------------------------------------------------------
        grid = ResponsiveGrid(spacing=14, min_item_width=340)

        input_card = Card(padding=14, spacing=10)
        input_card.body().addWidget(
            SectionHeader("Imagen de entrada", "Arrastra y suelta o elige un archivo")
        )
        self.drop = DropZone()
        self.drop.fileDropped.connect(self.set_image)
        self.drop.chooseRequested.connect(self.choose_image)
        input_card.body().addWidget(self.drop)

        net_row = QHBoxLayout()
        net_row.setSpacing(10)
        self.net_view = ImageView("Entrada de la CNN", "grid", min_height=86, rounded=8)
        self.net_view.setFixedWidth(112)
        net_row.addWidget(self.net_view)
        net_col = QVBoxLayout()
        net_col.setSpacing(2)
        net_title = QLabel("Lo que ve la red")
        net_title.setProperty("semibold", True)
        net_col.addWidget(net_title)
        self.net_info = QLabel("Tensor 3 × N × N · RGB normalizado")
        self.net_info.setProperty("subtle", True)
        self.net_info.setWordWrap(True)
        self.net_info.setStyleSheet("font-size: 10px;")
        net_col.addWidget(self.net_info)
        net_row.addLayout(net_col, 1)
        input_card.body().addLayout(net_row)
        grid.add(input_card)

        result_card = Card(padding=16, spacing=10)
        result_card.body().addWidget(
            SectionHeader("Resultado", "Probabilidades Softmax sobre las 3 clases")
        )
        self.pred_name = QLabel("Sin predicción")
        self.pred_name.setStyleSheet(f"color: {MUTED}; font-size: 28px; font-weight: 700;")
        self.pred_name.setWordWrap(True)
        result_card.body().addWidget(self.pred_name)
        self.pred_conf = QLabel("Selecciona una imagen para comenzar.")
        self.pred_conf.setProperty("muted", True)
        self.pred_conf.setWordWrap(True)
        result_card.body().addWidget(self.pred_conf)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setVisible(False)
        self.warning.setStyleSheet(
            f"color: {WARNING}; background: {rgba(WARNING, 0.08)};"
            f" border: 1px solid {rgba(WARNING, 0.30)}; border-radius: 8px;"
            " padding: 7px 9px; font-size: 10px;"
        )
        result_card.body().addWidget(self.warning)

        for cls in CLASSES:
            row = QHBoxLayout()
            row.setSpacing(8)
            name = QLabel(CLASS_LABELS[cls])
            name.setStyleSheet(f"color: {CLASS_COLORS[cls]}; font-size: 11px; font-weight: 600;")
            name.setMinimumWidth(58)
            row.addWidget(name)
            bar = _bar(CLASS_COLORS[cls])
            row.addWidget(bar, 1)
            pct = QLabel("—")
            pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct.setMinimumWidth(50)
            pct.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
            row.addWidget(pct)
            self._bars[cls] = bar
            self._pcts[cls] = pct
            self._names[cls] = name
            result_card.body().addLayout(row)

        self.detail = QLabel("")
        self.detail.setProperty("subtle", True)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("font-size: 10px;")
        result_card.body().addWidget(self.detail)

        guide = QLabel(
            "Lectura: ≥ 70 % alta confianza · 50–70 % conviene revisar · "
            "< 50 % baja confianza."
        )
        guide.setProperty("subtle", True)
        guide.setWordWrap(True)
        guide.setStyleSheet("font-size: 10px;")
        result_card.body().addWidget(guide)
        result_card.body().addStretch(1)
        grid.add(result_card)

        layout.addWidget(grid)
        layout.addStretch(1)

        self.refresh_model()

    # ------------------------------------------------------------------
    def refresh_model(self) -> None:
        model, cfg, label = self._model_and_cfg()
        if model is None:
            self.model_badge.setText("Modelo no disponible")
            self.model_badge.setStyleSheet(
                f"background: rgba(255,255,255,0.035); color: {MUTED};"
                f" border: 1px solid {rgba(MUTED, 0.33)}; border-radius: 10px;"
                " padding: 2px 9px; font-size: 11px; font-weight: 600;"
            )
        else:
            self.model_badge.setText(label)
            self.model_badge.setStyleSheet(
                f"background: rgba(255,255,255,0.035); color: {SUCCESS};"
                f" border: 1px solid {rgba(SUCCESS, 0.33)}; border-radius: 10px;"
                " padding: 2px 9px; font-size: 11px; font-weight: 600;"
            )
        if cfg is not None:
            size = int(cfg.input_size)
            self.net_info.setText(
                f"Tensor 3 × {size} × {size} · RGB normalizado · Otsu "
                f"{'activo' if cfg.roi_crop else 'desactivado'}"
            )

    def _model_and_cfg(self):
        main = self.main
        model = None
        cfg = None
        label = "Modelo incluido"
        # El modelo entrenado en esta sesión tiene prioridad, igual que en Resultados.
        result = getattr(main, "training_result", None)
        if result is not None and getattr(result, "model", None) is not None:
            model = result.model
            cfg = getattr(result, "config", None)
            acc = (getattr(result, "metrics", {}) or {}).get("accuracy")
            label = "Modelo de esta sesión"
            if acc is not None:
                label = f"Modelo de esta sesión · {float(acc) * 100:.1f}% test"
        if model is None:
            model = getattr(main, "loaded_model", None)
            cfg = getattr(main, "loaded_cfg", None)
            meta = getattr(main, "loaded_meta", {}) or {}
            acc = (meta.get("extra") or {}).get("accuracy")
            if acc is not None and model is not None:
                label = f"Modelo incluido · {float(acc) * 100:.1f}% test"
        if cfg is None:
            cfg = getattr(main, "training_cfg", None) or TrainingConfig()
        return model, cfg, label

    # ------------------------------------------------------------------
    def choose_image(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Elegir imagen para predecir", str(Path.home()),
            f"Imágenes ({patterns});;Todos los archivos (*)",
        )
        if path:
            self.set_image(path)

    def set_image(self, path: str) -> None:
        file = Path(path)
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            self._toast("Formato no compatible. Usa JPG, PNG, BMP, WEBP o TIFF.", "warning")
            return
        try:
            rgb = imread_rgb(file)
        except Exception as exc:  # noqa: BLE001
            self._toast(f"No se pudo leer la imagen: {exc}", "error")
            return
        if rgb is None or getattr(rgb, "size", 0) == 0:
            self._toast("La imagen está vacía o no se pudo decodificar.", "error")
            return
        self.path = str(file)
        self.file_label.setText(str(file))
        self.file_label.setToolTip(str(file))
        self.drop.view.set_image(rgb)
        self.repeat_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self._predict()

    def clear(self) -> None:
        self.path = None
        self._probs = None
        self.file_label.setText("Ninguna imagen seleccionada")
        self.drop.view.clear()
        self.net_view.clear()
        self.net_info.setText("Tensor 3 × N × N · RGB normalizado")
        self.pred_name.setText("Sin predicción")
        self.pred_name.setStyleSheet(f"color: {MUTED}; font-size: 28px; font-weight: 700;")
        self.pred_conf.setText("Selecciona una imagen para comenzar.")
        self.warning.setVisible(False)
        self.detail.setText("")
        for cls in CLASSES:
            self._bars[cls].setValue(0)
            self._pcts[cls].setText("—")
        self.repeat_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

    # ------------------------------------------------------------------
    def _predict(self) -> None:
        if not self.path:
            return
        model, cfg, _label = self._model_and_cfg()
        if model is None:
            self.pred_name.setText("Sin modelo")
            self.pred_name.setStyleSheet(f"color: {WARNING}; font-size: 24px; font-weight: 700;")
            self.pred_conf.setText(
                "Entrena la red en la pestaña Entrenamiento o carga un modelo guardado."
            )
            return
        try:
            tensor = preprocess_for_training(
                self.path,
                int(cfg.input_size),
                cfg.equalization,
                roi=bool(cfg.roi_crop),
                roi_margin=float(cfg.roi_margin),
                clahe_clip=float(getattr(cfg, "clahe_clip", 2.0)),
                clahe_tiles=int(getattr(cfg, "clahe_tiles", 8)),
            )
            probs = np.asarray(model.predict_proba(tensor[None, ...]))[0]
        except Exception as exc:  # noqa: BLE001
            self._toast(f"No se pudo predecir: {exc}", "error")
            return

        self._probs = probs
        idx = int(np.argmax(probs))
        cls = CLASSES[idx]
        conf = float(probs[idx]) * 100.0
        color = CLASS_COLORS[cls]

        self.pred_name.setText(CLASS_LABELS[cls])
        self.pred_name.setStyleSheet(f"color: {color}; font-size: 30px; font-weight: 700;")
        self.pred_conf.setText(f"Confianza: {conf:.1f} % · clase {idx + 1} de 3")
        self.warning.setVisible(conf < 50.0)
        if conf < 50.0:
            self.warning.setText(
                "Confianza baja: la imagen puede no parecerse a las fotos del dataset. "
                "Prueba con otra foto o revisa la segmentación."
            )

        for i, c in enumerate(CLASSES):
            p = float(probs[i])
            self._bars[c].setValue(int(round(p * 1000)))
            self._pcts[c].setText(f"{p * 100:.1f} %")
            weight = "700" if i == idx else "500"
            self._names[c].setStyleSheet(
                f"color: {CLASS_COLORS[c]}; font-size: 11px; font-weight: {weight};"
            )

        # Vista real de la entrada que consume la CNN (RGB en [0, 1]).
        rgb = np.clip(np.transpose(tensor, (1, 2, 0)), 0.0, 1.0)
        self.net_view.set_image((rgb * 255).astype(np.uint8))
        self.net_info.setText(
            f"Tensor 3 × {tensor.shape[1]} × {tensor.shape[2]} · RGB normalizado · "
            f"Otsu {'activo' if cfg.roi_crop else 'desactivado'}"
        )
        self.detail.setText(
            f"Salida de las 3 neuronas tras Softmax · suma {float(probs.sum()):.4f} · "
            f"archivo {Path(self.path).name}"
        )
        self.repeat_btn.setEnabled(True)

    def _toast(self, text: str, kind: str = "warning") -> None:
        if hasattr(self.main, "toast"):
            self.main.toast(text, kind)

    def showEvent(self, event) -> None:  # noqa: N802
        self.refresh_model()
        super().showEvent(event)
