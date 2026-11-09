"""Arquitectura viva de la red y laboratorio numérico de convolución/pooling.

Este módulo no depende de la interfaz: calcula formas, parámetros, operaciones
y ejemplos paso a paso que luego la página *Arquitectura* muestra en tablas.

Todo el cálculo es NumPy puro; no interviene ninguna librería de aprendizaje.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Utilidades de formato
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Utilidades de formato y constantes
# ---------------------------------------------------------------------------
EXCEL_COLUMNS: list[str] = [
    "capa",
    "Entrada (H×W×C)",
    "Kernel / ventana",
    "Stride",
    "Padding",
    "Filtros",
    "Salida (H×W×C)",
    "TOTAL PARAMETROS",
]


def fmt_int(value: int | float) -> str:
    """Entero con separador de miles (estilo es-CO: punto)."""
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def fmt_shape(shape: tuple[int, ...] | list[int] | None) -> str:
    if not shape:
        return "—"
    return " × ".join(str(int(v)) for v in shape)


@dataclass
class ArchRow:
    """Una fila de la tabla de arquitectura, compatible con la estructura del Excel de clase."""

    stage: str
    operation: str
    input_shape: tuple[int, ...] | None
    kernel: str = "—"
    stride: int | str | None = None
    padding: int | str | None = None
    filters: int | str | None = None
    output_shape: tuple[int, ...] | None = None
    elements: int = 0
    params: int = 0
    flops: int = 0
    formula: str = ""
    note: str = ""
    color: str = "#22d3ee"
    # Campos específicos del Excel de clase (hoja 'trabajo'):
    capa: str = ""
    entrada_excel: int | str = ""
    salida_excel: str = ""
    total_parametros: int | str = 0
    is_blue: bool = True
    formula_excel: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "operation": self.operation,
            "input": fmt_shape(self.input_shape),
            "kernel": self.kernel,
            "stride": "—" if self.stride in (None, "") else self.stride,
            "padding": "—" if self.padding in (None, "") else self.padding,
            "filters": "—" if self.filters in (None, "") else self.filters,
            "output": fmt_shape(self.output_shape),
            "elements": fmt_int(self.elements),
            "params": fmt_int(self.params),
            "flops": fmt_int(self.flops),
            "formula": self.formula,
            "note": self.note,
            "capa": self.capa or self.stage,
            "entrada_excel": self.entrada_excel,
            "salida_excel": self.salida_excel,
            "total_parametros": self.total_parametros,
            "is_blue": self.is_blue,
            "formula_excel": self.formula_excel,
        }

    def excel_row_values(self) -> list[str]:
        """Devuelve los valores correspondientes exactamente a las 8 columnas de la tabla."""
        return [
            str(self.capa or self.stage),
            str(self.entrada_excel) if self.entrada_excel != "" else "—",
            str(self.kernel) if self.kernel not in ("", "—") else "—",
            str(self.stride) if self.stride not in (None, "", "—") else "—",
            str(self.padding) if self.padding not in (None, "", "—") else "—",
            str(self.filters) if self.filters not in (None, "", "—") else "—",
            str(self.salida_excel) if self.salida_excel != "" else "—",
            fmt_int(self.total_parametros) if self.total_parametros != "" else "—",
        ]


def _out_size(size: int, kernel: int, padding: int, stride: int) -> int:
    return max(1, (int(size) + 2 * int(padding) - int(kernel)) // max(1, int(stride)) + 1)


def cnn_architecture_rows(
    input_size: int = 48,
    num_blocks: int = 2,
    in_channels: int = 1,
    filters_per_block: tuple[int, ...] | list[int] = (6, 12, 24),
    kernel_sizes: tuple[int, ...] | list[int] = (3, 3, 3),
    strides: tuple[int, ...] | list[int] = (1, 1, 1),
    paddings: tuple[int, ...] | list[int] = (1, 1, 1),
    pool_sizes: tuple[int, ...] | list[int] = (2, 2, 2),
    pool_strides: tuple[int, ...] | list[int] = (2, 2, 2),
    pools_enabled: tuple[bool, ...] | list[bool] = (True, True, True),
    hidden_units: int = 64,
    num_classes: int = 3,
) -> list[ArchRow]:
    """Genera el desglose capa por capa de la Red Neuronal Convolucional (CNN para Peces).

    Sigue estrictamente el flujo de tensores: la SALIDA de cada etapa se convierte
    en la ENTRADA exacta de la etapa siguiente.

    Columnas:
      capa | Entrada (H×W×C) | Kernel / ventana | Stride | Padding | Filtros | Salida (H×W×C) | TOTAL PARAMETROS

    Todos los valores son matemáticamente reales calculados con NumPy puro:
      H_out = floor((H_in + 2*P - K) / S) + 1
      W_out = floor((W_in + 2*P - K) / S) + 1
      Conv_total = H_out * W_out * Filtros
      Pool_total = HP_out * WP_out * Filtros
      FC_total = (H_out_dense * In_units) + H_out_dense
    """
    blocks = max(1, min(3, int(num_blocks)))
    size = max(8, int(input_size))
    in_ch = max(1, int(in_channels))
    hid = max(4, int(hidden_units))
    n_out = max(2, int(num_classes))

    filts = list(filters_per_block) + [24, 48, 64]
    ksizes = list(kernel_sizes) + [3, 3, 3]
    s_list = list(strides) + [1, 1, 1]
    p_list = list(paddings) + [1, 1, 1]
    psizes = list(pool_sizes) + [2, 2, 2]
    pstrides = list(pool_strides) + [2, 2, 2]
    penabled = list(pools_enabled) + [True, True, True]

    rows: list[ArchRow] = []

    # 1. IMAGEN DE ENTRADA (Preprocesada del pipeline de peces)
    img_tot = size * size * in_ch
    in_repr = f"{size} × {size} × {in_ch}  ({fmt_int(img_tot)})"
    out_repr = f"{size} × {size} × {in_ch}  ({fmt_int(img_tot)})"

    rows.append(ArchRow(
        stage="Entrada",
        capa="IMAGEN",
        operation="Imagen de entrada preprocesada",
        input_shape=(size, size, in_ch),
        output_shape=(size, size, in_ch),
        elements=img_tot,
        params=0,
        flops=0,
        entrada_excel=in_repr,
        salida_excel=out_repr,
        total_parametros=img_tot,
        is_blue=True,
        formula_excel=f"={size}*{size}*{in_ch}",
        note=f"Tensor que recibe la red convolucional ({size}×{size}×{in_ch}).",
        color="#38bdf8",
    ))

    # El tensor de salida de la imagen es la entrada de la primera convolución
    prev_tot = img_tot
    cur_h, cur_w = size, size
    cur_ch = in_ch
    prev_out_repr = out_repr

    # 2. Bloques Convolucionales + Pooling
    for b in range(1, blocks + 1):
        k = max(1, int(ksizes[b - 1]))
        s = max(1, int(s_list[b - 1]))
        p = max(0, int(p_list[b - 1]))
        f = max(1, int(filts[b - 1]))

        h_out = max(1, (cur_h + 2 * p - k) // s + 1)
        w_out = max(1, (cur_w + 2 * p - k) // s + 1)
        conv_tot = h_out * w_out * f

        # Parámetros entrenables y FLOPs reales
        learnable = (k * k * cur_ch + 1) * f
        flops_conv = h_out * w_out * learnable
        conv_out_repr = f"{h_out} × {w_out} × {f}  ({fmt_int(conv_tot)})"

        rows.append(ArchRow(
            stage=f"Conv {b}",
            capa=f"Conv {b}",
            operation=f"Convolución 2D ({k}×{k}) + activación ReLU",
            input_shape=(cur_h, cur_w, cur_ch),
            kernel=f"{k} × {k}",
            stride=s,
            padding=p,
            filters=f,
            output_shape=(h_out, w_out, f),
            elements=conv_tot,
            params=learnable,
            flops=flops_conv,
            entrada_excel=prev_out_repr,
            salida_excel=conv_out_repr,
            total_parametros=conv_tot,
            is_blue=True,
            formula=f"({k}·{k}·{cur_ch}+1)·{f} = {fmt_int(learnable)} pesos",
            formula_excel=f"={h_out}*{w_out}*{f}",
            note=f"Aprende {f} mapas de características sobre peces.",
            color="#22d3ee",
        ))

        # La salida de la convolución se vuelve la entrada del siguiente paso
        prev_tot = conv_tot
        cur_h, cur_w, cur_ch = h_out, w_out, f
        prev_out_repr = conv_out_repr

        # Pooling (si está habilitado para este bloque)
        if bool(penabled[b - 1]):
            pk = max(1, int(psizes[b - 1]))
            ps = max(1, int(pstrides[b - 1]))
            hp_out = max(1, (cur_h - pk) // ps + 1)
            wp_out = max(1, (cur_w - pk) // ps + 1)
            pool_tot = hp_out * wp_out * cur_ch
            pool_out_repr = f"{hp_out} × {wp_out} × {cur_ch}  ({fmt_int(pool_tot)})"

            rows.append(ArchRow(
                stage=f"Pool {b}",
                capa=f"Pool {b}",
                operation=f"Max Pooling ({pk}×{pk})",
                input_shape=(cur_h, cur_w, cur_ch),
                kernel=f"{pk} × {pk}",
                stride=ps,
                padding=0,
                filters=cur_ch,
                output_shape=(hp_out, wp_out, cur_ch),
                elements=pool_tot,
                params=0,
                flops=0,
                entrada_excel=prev_out_repr,
                salida_excel=pool_out_repr,
                total_parametros=pool_tot,
                is_blue=True,
                formula=f"⌊({cur_h}-{pk})/{ps}⌋+1 = {hp_out}",
                formula_excel=f"={hp_out}*{wp_out}*{cur_ch}",
                note=f"Reduce la resolución espacial a {hp_out}×{wp_out}.",
                color="#a78bfa",
            ))

            prev_tot = pool_tot
            cur_h, cur_w = hp_out, wp_out
            prev_out_repr = pool_out_repr

    # 3. Vectorización (Flatten)
    flat_tot = prev_tot
    flat_out_repr = f"1 × {fmt_int(flat_tot)}  ({fmt_int(flat_tot)})"
    rows.append(ArchRow(
        stage="Flatten",
        capa="Flatten",
        operation="Vectorización (Aplanado a 1D)",
        input_shape=(cur_h, cur_w, cur_ch),
        output_shape=(flat_tot,),
        elements=flat_tot,
        params=0,
        flops=0,
        entrada_excel=prev_out_repr,
        salida_excel=flat_out_repr,
        total_parametros=flat_tot,
        is_blue=False,
        formula_excel=f"1X{flat_tot}",
        note="Convierte los mapas 2D en un único vector 1D.",
        color="#f472b6",
    ))
    prev_out_repr = flat_out_repr

    # 4. Capa Totalmente Conectada (Densa / FC)
    fc_params = hid * flat_tot + hid
    fc_out_repr = f"{hid} neuronas  ({hid})"
    rows.append(ArchRow(
        stage="Densa 1",
        capa="Densa (FC)",
        operation=f"Capa Densa Oculta ({hid} neuronas con ReLU)",
        input_shape=(flat_tot,),
        output_shape=(hid,),
        elements=hid,
        params=fc_params,
        flops=fc_params,
        entrada_excel=prev_out_repr,
        salida_excel=fc_out_repr,
        total_parametros=fc_params,
        is_blue=False,
        formula=f"({flat_tot} × {hid}) + {hid} = {fmt_int(fc_params)} parámetros",
        formula_excel=f"=({hid}*{flat_tot})+{hid}",
        note=f"{hid} neuronas densas combinando patrones de peces.",
        color="#fbbf24",
    ))
    prev_out_repr = fc_out_repr

    # 5. Capa de Salida Softmax (Clasificación de los 3 peces)
    sm_params = n_out * hid + n_out
    sm_out_repr = f"{n_out} clases (Trucha, Mojarra, Pavón)"
    rows.append(ArchRow(
        stage="Softmax",
        capa="Softmax",
        operation="Capa de Salida con Probabilidades Softmax",
        input_shape=(hid,),
        output_shape=(n_out,),
        elements=n_out,
        params=sm_params,
        flops=sm_params,
        entrada_excel=prev_out_repr,
        salida_excel=sm_out_repr,
        total_parametros=sm_params,
        is_blue=False,
        formula=f"({hid} × {n_out}) + {n_out} = {fmt_int(sm_params)} parámetros",
        formula_excel=f"=({n_out}*{hid})+{n_out}",
        note="Distribución de probabilidad entre Trucha, Mojarra y Pavón.",
        color="#34d399",
    ))

    return rows


# Alias para compatibilidad
excel_architecture_rows = cnn_architecture_rows


def cnn_rows(
    input_size: int = 48,
    filters1: int = 6,
    filters2: int = 12,
    hidden: int = 64,
    n_out: int = 3,
    kernel: int = 3,
    padding: int = 1,
    pool: int = 2,
    in_ch: int = 1,
) -> list[ArchRow]:
    """Arquitectura viva de la CNN para clasificación de peces (NumPy puro)."""
    return cnn_architecture_rows(
        input_size=int(input_size),
        num_blocks=2,
        in_channels=int(in_ch),
        filters_per_block=(int(filters1), int(filters2)),
        kernel_sizes=(int(kernel), int(kernel)),
        strides=(1, 1),
        paddings=(int(padding), int(padding)),
        pool_sizes=(int(pool), int(pool)),
        pool_strides=(int(pool), int(pool)),
        pools_enabled=(True, True),
        hidden_units=int(hidden),
        num_classes=int(n_out),
    )


def totals(rows: list[ArchRow]) -> dict[str, int]:
    return {
        "params": sum(int(r.params) for r in rows),
        "elements": sum(int(r.elements) for r in rows),
        "flops": sum(int(r.flops) for r in rows),
        "layers": sum(1 for r in rows if r.params > 0),
    }


# ---------------------------------------------------------------------------
# Laboratorio numérico: convolución y pooling paso a paso
# ---------------------------------------------------------------------------
def as_matrix(values: Any, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(rows, cols)
    if arr.shape != (rows, cols):
        fixed = np.zeros((rows, cols), np.float32)
        r = min(rows, arr.shape[0])
        c = min(cols, arr.shape[1])
        fixed[:r, :c] = arr[:r, :c]
        return fixed
    return arr


def conv_demo(
    matrix: Any,
    kernel: Any,
    padding: int = 0,
    stride: int = 1,
) -> dict[str, Any]:
    """Convolución 2D con padding/stride, devolviendo también las ventanas."""
    x = np.asarray(matrix, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    if k.shape != (3, 3):
        k = np.resize(k, (3, 3)).astype(np.float32)
    p = max(0, int(padding))
    s = max(1, int(stride))
    xp = np.pad(x, ((p, p), (p, p)), mode="constant") if p else x
    h, w = xp.shape
    if h < 3 or w < 3:
        return {"output": np.zeros((1, 1), np.float32), "padded": xp, "starts": [(0, 0)], "shape": (1, 1)}
    hout = (h - 3) // s + 1
    wout = (w - 3) // s + 1
    out = np.zeros((hout, wout), np.float32)
    starts: list[tuple[int, int]] = []
    for i in range(hout):
        for j in range(wout):
            y, xx = i * s, j * s
            out[i, j] = float(np.sum(xp[y:y + 3, xx:xx + 3] * k))
            starts.append((y, xx))
    return {"output": out, "padded": xp, "starts": starts, "shape": (hout, wout)}


def conv_step_detail(
    matrix: Any,
    kernel: Any,
    padding: int = 0,
    stride: int = 1,
    row: int = 0,
    col: int = 0,
) -> dict[str, Any]:
    """Detalle de una celda: ventana, productos, suma y posición."""
    x = np.asarray(matrix, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    p = max(0, int(padding))
    s = max(1, int(stride))
    xp = np.pad(x, ((p, p), (p, p)), mode="constant") if p else x
    y, xx = int(row) * s, int(col) * s
    window = xp[y:y + 3, xx:xx + 3]
    if window.shape != (3, 3):
        window = np.zeros((3, 3), np.float32)
    products = window * k
    return {
        "row": int(row),
        "col": int(col),
        "origin": (y, xx),
        "window": window,
        "kernel": k,
        "products": products,
        "sum": float(products.sum()),
        "formula": " + ".join(
            f"({window[i, j]:g}·{k[i, j]:g})" for i in range(3) for j in range(3)
        ) + f" = {products.sum():g}",
    }


def pool_demo(
    matrix: Any,
    mode: str = "max",
    size: int = 2,
    stride: int = 2,
) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=np.float32)
    k = max(1, int(size))
    s = max(1, int(stride))
    h, w = x.shape
    if h < k or w < k:
        return {"output": x.copy(), "starts": [(0, 0)], "shape": x.shape}
    hout = (h - k) // s + 1
    wout = (w - k) // s + 1
    out = np.zeros((hout, wout), np.float32)
    starts: list[tuple[int, int]] = []
    for i in range(hout):
        for j in range(wout):
            y, xx = i * s, j * s
            win = x[y:y + k, xx:xx + k]
            out[i, j] = float(win.max() if mode == "max" else win.mean())
            starts.append((y, xx))
    return {"output": out, "starts": starts, "shape": (hout, wout)}


def pool_step_detail(
    matrix: Any,
    mode: str = "max",
    size: int = 2,
    stride: int = 2,
    row: int = 0,
    col: int = 0,
) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=np.float32)
    k = max(1, int(size))
    s = max(1, int(stride))
    y, xx = int(row) * s, int(col) * s
    win = x[y:y + k, xx:xx + k]
    value = float(win.max() if mode == "max" else win.mean())
    op = "max" if mode == "max" else "media"
    values = ", ".join(f"{v:g}" for v in win.ravel())
    return {
        "row": int(row), "col": int(col), "window": win, "value": value,
        "formula": f"{op}({values}) = {value:g}",
    }


def sample_matrix(size: int = 10, seed: int = 7) -> np.ndarray:
    """Matriz de ejemplo con valores enteros de 0 a 255."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, size, dtype=np.float32)
    grid = np.add.outer(base, base[::-1]) / 2.0
    noise = rng.normal(0, 18, (size, size))
    return np.clip(grid + noise, 0, 255).round().astype(np.float32)


def matrix_from_patch(image: np.ndarray, size: int = 10) -> np.ndarray:
    """Extrae un parche central de una imagen y lo reescala a ``size``×``size``."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    h, w = arr.shape[:2]
    side = max(1, min(h, w, size * 4))
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    patch = arr[y0:y0 + side, x0:x0 + side]
    if patch.size == 0:
        return np.zeros((size, size), np.float32)
    # Rejilla por bloques para conservar el aspecto de píxeles.
    ys = np.linspace(0, patch.shape[0], size + 1).astype(int)
    xs = np.linspace(0, patch.shape[1], size + 1).astype(int)
    out = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            block = patch[ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return np.clip(out, 0, 255).round()


# ---------------------------------------------------------------------------
# Descripción textual usada por la auditoría y la interfaz
# ---------------------------------------------------------------------------
MANUAL_ENGINE_CHECKS = (
    ("Lectura de imágenes", "OpenCV solo decodifica píxeles; no aprende nada."),
    ("Convolución", "core/image_ops.convolve2d: ventanas + producto punto en NumPy."),
    ("Forward CNN", "core/cnn.py: Conv2D, ReLU, MaxPool2D y Flatten propios."),
    ("Backpropagation", "core/cnn.py: gradientes analíticos de conv, pooling y densas."),
    ("Softmax", "core/nn.py: exp(z − max) / suma, escrito a mano."),
    ("Entropía cruzada", "core/nn.py: −log(p[y]) con pesos por clase opcionales."),
    ("Momentum", "core/nn.py: v ← μv − η∇L ; θ ← θ + v."),
    ("Inicialización", "He/Xavier con np.random.Generator (sin frameworks)."),
)
FORBIDDEN_LEARNING_LIBS = (
    "tensorflow", "torch", "keras", "sklearn", "scikit-learn", "jax",
    "theano", "mxnet", "caffe", "paddle", "onnxruntime", "autograd",
)


def audit_imports(root: str | None = None) -> dict[str, Any]:
    """Recorre ``core/`` y comprueba que no se importe un framework de DL."""
    from pathlib import Path

    base = Path(root) if root else Path(__file__).resolve().parent
    hits: list[str] = []
    files = 0
    for path in sorted(base.glob("*.py")):
        files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        for lib in FORBIDDEN_LEARNING_LIBS:
            if f"import {lib}" in text or f"from {lib}" in text:
                hits.append(f"{path.name}: {lib}")
    return {"files": files, "hits": hits, "ok": not hits}


# ---------------------------------------------------------------------------
# Recorrido paso a paso: kernel por kernel, celda por celda, hasta el pooling
# ---------------------------------------------------------------------------
@dataclass
class LabStep:
    """Un fotograma del recorrido numérico de convolución + ReLU + pooling."""

    index: int
    phase: str                 # conv | relu | pool | done
    title: str
    detail: str
    source: np.ndarray
    source_highlight: tuple[int, int, int, int] | None
    kernel: np.ndarray | None
    products: np.ndarray | None
    target: np.ndarray
    target_highlight: tuple[int, int, int, int] | None
    formula: str
    value: float | None
    kernel_index: int
    kernel_total: int
    cell_index: int
    cell_total: int
    source_label: str = ""
    target_label: str = ""


def _prepare_kernel(kernel: Any) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32)
    if k.shape != (3, 3):
        k = np.resize(k, (3, 3)).astype(np.float32)
    return k


def build_lab_steps(
    matrix: Any,
    kernels: list[Any] | tuple[Any, ...] | None = None,
    padding: int = 0,
    stride: int = 1,
    relu: bool = True,
    pool_mode: str = "max",
    pool_size: int = 2,
    pool_stride: int = 2,
) -> list[LabStep]:
    """Genera todos los pasos del laboratorio, en orden.

    Fase 1: cada kernel recorre la imagen celda por celda (ventana × kernel).
    Fase 2: ReLU sobre cada mapa de características.
    Fase 3: pooling celda por celda sobre cada mapa.
    Fase 4: resultado final con las matrices transformadas.
    """
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2:
        x = x.reshape(x.shape[-2], x.shape[-1]) if x.ndim > 2 else x.reshape(1, -1)
    if kernels is None:
        kernels = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    ks = [_prepare_kernel(k) for k in kernels] or [_prepare_kernel([[0]])]
    p = max(0, int(padding))
    s = max(1, int(stride))
    xp = np.pad(x, ((p, p), (p, p)), mode="constant") if p else x
    h, w = xp.shape
    hout = max(1, (h - 3) // s + 1)
    wout = max(1, (w - 3) // s + 1)
    cells = hout * wout
    ktotal = len(ks)
    steps: list[LabStep] = []
    raw_maps: list[np.ndarray] = []

    for ki, k in enumerate(ks):
        out = np.zeros((hout, wout), np.float32)
        for i in range(hout):
            for j in range(wout):
                y, xx = i * s, j * s
                window = xp[y:y + 3, xx:xx + 3]
                if window.shape != (3, 3):
                    window = np.zeros((3, 3), np.float32)
                products = window * k
                value = float(products.sum())
                out[i, j] = value
                steps.append(LabStep(
                    index=len(steps),
                    phase="conv",
                    title=f"Kernel {ki + 1}/{ktotal} · celda {i * wout + j + 1}/{cells}",
                    detail=(
                        f"La ventana 3×3 se coloca en la fila {y}, columna {xx} de la entrada "
                        f"{'(con padding)' if p else ''}. Se multiplica elemento a elemento por el "
                        f"kernel y se suman los 9 productos."
                    ),
                    source=xp.copy(),
                    source_highlight=(y, xx, 3, 3),
                    kernel=k.copy(),
                    products=products.copy(),
                    target=out.copy(),
                    target_highlight=(i, j, 1, 1),
                    formula=(
                        " + ".join(
                            f"({window[a, b]:g}·{k[a, b]:g})" for a in range(3) for b in range(3)
                        )
                        + f" = {value:g}"
                    ),
                    value=value,
                    kernel_index=ki,
                    kernel_total=ktotal,
                    cell_index=i * wout + j,
                    cell_total=cells,
                    source_label=f"Entrada {h}×{w}" + (" (padding)" if p else ""),
                    target_label=f"Mapa del kernel {ki + 1}",
                ))
        raw_maps.append(out)

    activated: list[np.ndarray] = []
    if relu:
        for ki, raw in enumerate(raw_maps):
            act = np.maximum(0.0, raw)
            negatives = int((raw < 0).sum())
            steps.append(LabStep(
                index=len(steps),
                phase="relu",
                title=f"ReLU · mapa {ki + 1}/{ktotal}",
                detail=(
                    f"Se aplica max(0, x) a las {raw.size} celdas del mapa. "
                    f"{negatives} valores negativos se convierten en 0; los positivos no cambian."
                ),
                source=raw.copy(),
                source_highlight=None,
                kernel=None,
                products=None,
                target=act.copy(),
                target_highlight=None,
                formula="y = max(0, x)",
                value=float(act.max()) if act.size else None,
                kernel_index=ki,
                kernel_total=ktotal,
                cell_index=0,
                cell_total=int(raw.size),
                source_label=f"Convolución {ki + 1}",
                target_label=f"ReLU {ki + 1}",
            ))
            activated.append(act)
    else:
        activated = [m.copy() for m in raw_maps]

    pooled_maps: list[np.ndarray] = []
    size = max(1, int(pool_size))
    ps = max(1, int(pool_stride))
    for ki, act in enumerate(activated):
        ah, aw = act.shape
        ph = max(1, (ah - size) // ps + 1) if ah >= size else ah
        pw = max(1, (aw - size) // ps + 1) if aw >= size else aw
        pooled = np.zeros((ph, pw), np.float32)
        total = ph * pw
        op = "max" if pool_mode == "max" else "media"
        for i in range(ph):
            for j in range(pw):
                y, xx = i * ps, j * ps
                win = act[y:y + size, xx:xx + size]
                if win.size == 0:
                    win = np.zeros((size, size), np.float32)
                value = float(win.max() if pool_mode == "max" else win.mean())
                pooled[i, j] = value
                vals = ", ".join(f"{v:g}" for v in win.ravel())
                steps.append(LabStep(
                    index=len(steps),
                    phase="pool",
                    title=f"Pooling · mapa {ki + 1}/{ktotal} · celda {i * pw + j + 1}/{total}",
                    detail=(
                        f"Ventana {size}×{size} en la fila {y}, columna {xx}. "
                        f"Se resume en un solo valor con la operación {op}."
                    ),
                    source=act.copy(),
                    source_highlight=(y, xx, size, size),
                    kernel=None,
                    products=None,
                    target=pooled.copy(),
                    target_highlight=(i, j, 1, 1),
                    formula=f"{op}({vals}) = {value:g}",
                    value=value,
                    kernel_index=ki,
                    kernel_total=ktotal,
                    cell_index=i * pw + j,
                    cell_total=total,
                    source_label=f"ReLU {ki + 1}",
                    target_label=f"Pooling {ki + 1}",
                ))
        pooled_maps.append(pooled)

    shapes = " · ".join(f"{m.shape[0]}×{m.shape[1]}" for m in pooled_maps)
    steps.append(LabStep(
        index=len(steps),
        phase="done",
        title="Resultado final del recorrido",
        detail=(
            f"{ktotal} kernel(s) aplicados sobre una entrada de {x.shape[0]}×{x.shape[1]}. "
            f"Cada kernel produjo su propio mapa; ReLU y pooling se aplicaron por separado. "
            f"Mapas finales: {shapes}. Esos mapas se aplanan y entran a la red densa."
        ),
        source=pooled_maps[0].copy() if pooled_maps else np.zeros((1, 1), np.float32),
        source_highlight=None,
        kernel=None,
        products=None,
        target=pooled_maps[-1].copy() if pooled_maps else np.zeros((1, 1), np.float32),
        target_highlight=None,
        formula=f"entrada {x.shape[0]}×{x.shape[1]} → {ktotal} mapa(s) {shapes}",
        value=None,
        kernel_index=ktotal - 1,
        kernel_total=ktotal,
        cell_index=0,
        cell_total=1,
        source_label="Mapa final 1",
        target_label=f"Mapa final {ktotal}",
    ))
    return steps
