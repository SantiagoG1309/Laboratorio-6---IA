"""Página de resultados: métricas, matriz de confusión, predicciones y exportación."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.config import CLASS_COLORS, CLASS_LABELS, CLASSES
from core.metrics import report_lines
from ui import icons
from ui.theme import (
    ACCENT,
    BORDER,
    DANGER,
    MUTED,
    ON_ACCENT,
    PURPLE,
    SUCCESS,
    SURFACE_2,
    TEXT,
    WARNING,
    rgba,
)
from ui.widgets.cards import Badge, Card, Divider, EmptyState, SectionHeader, StatCard
from ui.widgets.charts import ConfusionMatrixCanvas
from ui.widgets.flow_layout import ResponsiveGrid
from ui.widgets.image_view import ImageView, open_image_dialog
from ui.widgets.inspector import WeightInspector
from ui.workers import ExportThread, ThumbWorker


class PredictionCard(QWidget):
    def __init__(self, path: str, true_label: int, pred: int, confidence: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(190)
        self.path = path
        ok = true_label == pred
        color = SUCCESS if ok else DANGER
        card = Card(padding=8, spacing=5)
        card.setStyleSheet(
            f"QFrame[card='true'] {{ border: 1px solid {rgba(color, 0.4)}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(card)
        iv = ImageView("", "image", min_height=112, rounded=8)
        iv.setMinimumHeight(112)
        self.iv = iv
        iv.doubleClicked.connect(lambda p=path, t=true_label, pr=pred: open_image_dialog(
            self, _load(p), Path(p).name,
            f"Real: {CLASS_LABELS[CLASSES[t]]} · Predicción: {CLASS_LABELS[CLASSES[pr]]}",
        ))
        card.body().addWidget(iv)
        row = QHBoxLayout()
        row.setSpacing(4)
        true_b = Badge(CLASS_LABELS[CLASSES[true_label]], CLASS_COLORS[CLASSES[true_label]])
        pred_b = Badge(CLASS_LABELS[CLASSES[pred]], CLASS_COLORS[CLASSES[pred]])
        row.addWidget(true_b)
        arrow = QLabel("→")
        arrow.setStyleSheet(f"color: {MUTED};")
        row.addWidget(arrow)
        row.addWidget(pred_b)
        row.addStretch(1)
        card.body().addLayout(row)
        conf = QLabel(f"{confidence:.1f}% confianza")
        conf.setProperty("subtle", True)
        conf.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
        card.body().addWidget(conf)
        name = QLabel(Path(path).name[:26])
        name.setProperty("subtle", True)
        name.setStyleSheet("font-size: 9px;")
        card.body().addWidget(name)

    def set_thumb(self, thumb: str | None) -> None:
        if not thumb:
            return
        pm = QPixmap(str(thumb))
        if not pm.isNull():
            self.iv.set_image(pm)


def _load(path: str):
    from core.image_ops import imread_rgb
    try:
        return imread_rgb(path)
    except Exception:  # noqa: BLE001
        return None


class ResultsPage(QWidget):
    def __init__(self, main, parent: QWidget | None = None):
        super().__init__(parent)
        self.main = main
        self.result = None
        self._filter_mode = "all"
        self.export_thread: ExportThread | None = None
        self._progress: QProgressDialog | None = None
        self._pred_cards: list[PredictionCard] = []
        self._thumb_workers: list[ThumbWorker] = []
        self._thumb_token = 0

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

        # Empty
        self.empty = EmptyState(
            "chart",
            "Aún no hay resultados",
            "Entrena la red neuronal para ver la exactitud, la matriz de confusión y las predicciones "
            "sobre el conjunto de prueba.",
        )
        self.go_train = QPushButton("Ir a entrenamiento")
        self.go_train.setProperty("variant", "primary")
        self.go_train.setIcon(icons.icon("brain", ON_ACCENT, 16))
        self.go_train.clicked.connect(lambda: main.goto_page(3))
        self.empty.add_button(self.go_train)
        layout.addWidget(self.empty)

        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(16)
        layout.addWidget(self.body)

        # Header
        head = QHBoxLayout()
        hcol = QVBoxLayout()
        t = QLabel("Resultados del clasificador")
        t.setProperty("h1", True)
        hcol.addWidget(t)
        self.subtitle = QLabel("Métricas sobre el conjunto de prueba")
        self.subtitle.setProperty("muted", True)
        hcol.addWidget(self.subtitle)
        head.addLayout(hcol, 1)
        self.export_btn = QPushButton("Exportar dataset procesado")
        self.export_btn.setProperty("variant", "primary")
        self.export_btn.setIcon(icons.icon("export", ON_ACCENT, 16))
        self.report_btn = QPushButton("Reporte CSV")
        self.report_btn.setProperty("variant", "ghost")
        self.report_btn.setIcon(icons.icon("download", MUTED, 15))
        self.export_btn.setToolTip("Exporta las imágenes transformadas por etapa (original, canales, grises, convolución, pooling…).")
        self.report_btn.setToolTip("Genera un CSV con las métricas globales, por clase y la matriz de confusión.")
        head.addWidget(self.report_btn)
        head.addWidget(self.export_btn)
        bl.addLayout(head)

        # Stats
        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.stat_acc = StatCard("target", "Exactitud (test)", "—", SUCCESS)
        self.stat_f1 = StatCard("chart", "F1 macro", "—", ACCENT)
        self.stat_prec = StatCard("check", "Precisión macro", "—", PURPLE)
        self.stat_rec = StatCard("wave", "Recall macro", "—", WARNING)
        self.stat_best = StatCard("sparkles", "Mejor val acc.", "—", ACCENT)
        self.stat_acc.setToolTip("Exactitud (accuracy): porcentaje de imágenes de prueba clasificadas correctamente.")
        self.stat_f1.setToolTip("F1 macro: media armónica de precisión y recall, promediada por clase.")
        self.stat_prec.setToolTip("Precisión macro: de lo que la red predijo como una clase, cuánto era correcto.")
        self.stat_rec.setToolTip("Recall macro: de todas las imágenes reales de una clase, cuántas encontró la red.")
        self.stat_best.setToolTip("Mejor exactitud alcanzada sobre el conjunto de validación durante el entrenamiento.")
        for s in (self.stat_acc, self.stat_f1, self.stat_prec, self.stat_rec, self.stat_best):
            stats.addWidget(s, 1)
        bl.addLayout(stats)

        self.model_info = QLabel("")
        self.model_info.setProperty("muted", True)
        self.model_info.setStyleSheet(
            f"background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 10px;"
            "padding: 8px 12px; font-size: 11px;"
        )
        bl.addWidget(self.model_info)

        # CM + table
        mid_host = QWidget()
        mid_host.setFixedHeight(470)
        mid = QHBoxLayout(mid_host)
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(12)
        cm_card = Card(padding=12, spacing=4)
        cm_head = SectionHeader("Matriz de confusión", "Filas: clase real · columnas: predicción")
        cm_card.body().addWidget(cm_head)
        self.cm = ConfusionMatrixCanvas()
        self.cm.setMinimumHeight(340)
        cm_card.body().addWidget(self.cm)
        mid.addWidget(cm_card, 3)

        table_card = Card(padding=12, spacing=6)
        table_card.body().addWidget(SectionHeader("Métricas por clase", "Precisión · recall · F1"))
        self.table = QTableWidget(3, 5)
        self.table.setHorizontalHeaderLabels(["Clase", "Precisión", "Recall", "F1", "Soporte"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(220)
        table_card.body().addWidget(self.table)
        self.report_text = QLabel("")
        self.report_text.setProperty("mono", True)
        self.report_text.setWordWrap(True)
        self.report_text.setVisible(False)
        table_card.body().addWidget(self.report_text)
        table_card.body().addStretch(1)
        mid.addWidget(table_card, 2)
        bl.addWidget(mid_host)

        # Predictions
        pred_card = Card(padding=14, spacing=10)
        ph = SectionHeader("Predicciones del conjunto de prueba", "Doble clic para ampliar")
        self.filter_group = QButtonGroup(self)
        self.filter_buttons = {}
        for key, label in (("all", "Todas"), ("ok", "Aciertos"), ("err", "Errores")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("variant", "chip")
            b.clicked.connect(lambda _=False, k=key: self._set_filter(k))
            self.filter_group.addButton(b)
            self.filter_buttons[key] = b
            ph.add_right(b)
        self.filter_buttons["all"].setChecked(True)
        pred_card.body().addWidget(ph)
        self.pred_grid = ResponsiveGrid(spacing=10, min_item_width=208)
        pred_card.body().addWidget(self.pred_grid)
        bl.addWidget(pred_card)

        # Inspector de pesos y gradientes
        insp_card = Card(padding=14, spacing=10)
        insp_head = SectionHeader(
            "Inspector de pesos y gradientes",
            "Backpropagation desde cero · selecciona un tensor para ver su mapa de calor",
        )
        insp_card.body().addWidget(insp_head)
        self.inspector = WeightInspector()
        self.inspector.setMinimumHeight(430)
        insp_card.body().addWidget(self.inspector)
        bl.addWidget(insp_card)

        # Export info
        export_card = Card(padding=14, spacing=8)
        eh = SectionHeader("Exportar resultados", "Dataset procesado por etapas y reporte")
        export_card.body().addWidget(eh)
        info = QLabel(
            "La exportación guarda las carpetas 01_original, 02_canales_rgb_hsv, 03_escala_grises, "
            "04_roi_redimensionada, 05_ecualizado, 06_convolucion y 07_pooling, además de "
            "caracteristicas.csv, caracteristicas.npz y metadata.json."
        )
        info.setProperty("muted", True)
        info.setWordWrap(True)
        export_card.body().addWidget(info)
        bl.addWidget(export_card)

        self.export_btn.clicked.connect(self.export_dataset)
        self.report_btn.clicked.connect(self.export_report)
        self.body.setVisible(False)

    # ------------------------------------------------------------------
    def refresh(self, result) -> None:
        self.result = result
        if not result:
            self.empty.setVisible(True)
            self.body.setVisible(False)
            self.inspector.clear()
            return
        self.empty.setVisible(False)
        self.body.setVisible(True)
        m = result.metrics
        self.stat_acc.set_value(f"{m['accuracy'] * 100:.2f}%")
        self.stat_f1.set_value(f"{m['macro_f1'] * 100:.2f}%")
        self.stat_prec.set_value(f"{m['macro_precision'] * 100:.2f}%")
        self.stat_rec.set_value(f"{m['macro_recall'] * 100:.2f}%")
        self.stat_best.set_value(f"{m.get('best_val_accuracy', 0) * 100:.2f}%")
        cfg = result.config
        self.subtitle.setText(
            f"Modelo {cfg.model_type.upper()} · {result.test_count} imágenes de prueba · "
            f"{result.train_count} train / {result.val_count} val"
        )
        self.model_info.setText(
            f"Arquitectura: {cfg.model_type.upper()} · entrada {cfg.input_size}×{cfg.input_size} · "
            f"épocas {len(result.history.get('loss', []))} · mejor época {result.best_epoch} · "
            f"duración {result.duration:.1f}s · optimizador Momentum (lr={cfg.learning_rate}, μ={cfg.momentum})"
        )
        self.cm.update_cm(m["confusion"])
        self._fill_table(m)
        self._rebuild_predictions()
        self.inspector.set_model(result.model)

    def _fill_table(self, m: dict) -> None:
        for i, row in enumerate(m["per_class"]):
            vals = [
                CLASS_LABELS[CLASSES[i]],
                f"{row['precision'] * 100:.1f}%",
                f"{row['recall'] * 100:.1f}%",
                f"{row['f1'] * 100:.1f}%",
                str(row["support"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j == 0:
                    item.setForeground(QColor(CLASS_COLORS[CLASSES[i]]))
                self.table.setItem(i, j, item)
            self.table.setRowHeight(i, 34)
        self.report_text.setText("\n".join(report_lines(m, [CLASS_LABELS[c] for c in CLASSES])))

    def _set_filter(self, key: str) -> None:
        self._filter_mode = key
        for k, b in self.filter_buttons.items():
            b.setChecked(k == key)
        self._rebuild_predictions()

    def _rebuild_predictions(self) -> None:
        self._thumb_token += 1
        token = self._thumb_token
        self._pred_cards = []
        while self.pred_grid._flow.count():  # type: ignore[attr-defined]
            item = self.pred_grid._flow.takeAt(0)  # type: ignore[attr-defined]
            if item and item.widget():
                item.widget().deleteLater()
        if not self.result:
            return
        res = self.result
        for i, path in enumerate(res.test_paths):
            if i >= len(res.y_pred):
                break
            true = int(res.y_true[i]) if i < len(res.y_true) else int(res.test_labels[i])
            pred = int(res.y_pred[i])
            ok = true == pred
            if self._filter_mode == "ok" and not ok:
                continue
            if self._filter_mode == "err" and ok:
                continue
            conf = float(res.probs[i, pred] * 100) if i < len(res.probs) else 0.0
            card = PredictionCard(path, true, pred, conf)
            self._pred_cards.append(card)
            self.pred_grid.add(card)
        self._load_thumbs_async([c.path for c in self._pred_cards], token)

    def _load_thumbs_async(self, paths: list[str], token: int) -> None:
        if not paths:
            return
        worker = ThumbWorker(paths)
        self._thumb_workers.append(worker)
        worker.signals.done.connect(lambda thumbs, t=token: self._apply_thumbs(thumbs, t))
        worker.signals.done.connect(lambda *_: self.main._release_worker(self._thumb_workers, worker))
        self.main.thread_pool.start(worker)

    def _apply_thumbs(self, thumbs: dict, token: int) -> None:
        if token != self._thumb_token:
            return
        for card in self._pred_cards:
            card.set_thumb(thumbs.get(str(card.path)))

    def export_dataset(self) -> None:
        if not self.main.dataset_info or self.main.dataset_info.is_empty:
            self.main.toast("No hay dataset para exportar.", "warning")
            return
        default = str(self.main.exports_dir / f"Dataset_procesado_{datetime.now():%Y%m%d_%H%M}")
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de exportación", default)
        if not folder:
            return
        self._progress = QProgressDialog("Exportando etapas del pipeline…", "Cancelar", 0, 100, self)
        self._progress.setWindowTitle("Exportando dataset")
        self._progress.setMinimumWidth(420)
        self._progress.setValue(0)
        self.export_thread = ExportThread(
            self.main.dataset_info.samples,
            self.main.pipeline_cfg.copy(),
            folder,
            self,
        )
        self.export_thread.progress.connect(self._on_export_progress)
        self.export_thread.log.connect(lambda m: self.main.set_status(m))
        self.export_thread.done.connect(self._on_export_done)
        self.export_thread.failed.connect(self._on_export_failed)
        self._progress.canceled.connect(self.export_thread.request_stop)
        self.export_thread.start()
        self.main.toast("Exportación iniciada en segundo plano.", "info")

    def _on_export_progress(self, i: int, n: int, name: str) -> None:
        if self._progress:
            self._progress.setMaximum(n)
            self._progress.setValue(i)
            self._progress.setLabelText(f"Exportando {i}/{n}\n{name}")

    def _on_export_done(self, path: str) -> None:
        if self._progress:
            self._progress.setValue(self._progress.maximum())
            self._progress.close()
        self.main.toast(f"Dataset exportado en {path}", "success")
        self.main.set_status(f"Exportación completada: {path}")

    def _on_export_failed(self, msg: str) -> None:
        if self._progress:
            self._progress.close()
        self.main.toast("Error al exportar: " + msg.splitlines()[0], "error")

    def export_report(self) -> None:
        if not self.result:
            self.main.toast("No hay resultados para exportar.", "warning")
            return
        default = str(self.main.exports_dir / f"reporte_{datetime.now():%Y%m%d_%H%M}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Guardar reporte", default, "CSV (*.csv)")
        if not path:
            return
        res = self.result
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh, delimiter=";")
                w.writerow(["AquaCNN · Laboratorio 6 · Reporte de clasificación"])
                w.writerow(["Modelo", res.config.model_type.upper()])
                w.writerow(["Exactitud", f"{res.metrics['accuracy'] * 100:.2f}%"])
                w.writerow(["F1 macro", f"{res.metrics['macro_f1'] * 100:.2f}%"])
                w.writerow([])
                w.writerow(["Clase", "Precisión", "Recall", "F1", "Soporte"])
                for label, row in zip([CLASS_LABELS[c] for c in CLASSES], res.metrics["per_class"]):
                    w.writerow([label, f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                                f"{row['f1']:.3f}", row["support"]])
                w.writerow([])
                w.writerow(["Archivo", "Clase real", "Predicción", "Confianza"])
                for i, p in enumerate(res.test_paths):
                    if i >= len(res.y_pred):
                        break
                    true = int(res.y_true[i])
                    pred = int(res.y_pred[i])
                    conf = float(res.probs[i, pred]) if i < len(res.probs) else 0.0
                    w.writerow([Path(p).name, CLASS_LABELS[CLASSES[true]],
                                CLASS_LABELS[CLASSES[pred]], f"{conf:.4f}"])
            self.main.toast("Reporte CSV guardado.", "success")
        except Exception as exc:  # noqa: BLE001
            self.main.toast(f"No se pudo guardar el reporte: {exc}", "error")
