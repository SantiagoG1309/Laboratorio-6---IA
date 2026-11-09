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
      hidden_params = (unidades_entrada × neuronas_ocultas) + neuronas_ocultas
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

    # 4. Capa oculta de backpropagation (x·W + b → ReLU)
    hidden_params = hid * flat_tot + hid
    hidden_out_repr = f"{hid} neuronas  ({hid})"
    rows.append(ArchRow(
        stage="Oculta 1",
        capa="Oculta",
        operation=f"Capa oculta ({hid} neuronas, z = x·W + b → ReLU)",
        input_shape=(flat_tot,),
        output_shape=(hid,),
        elements=hid,
        params=hidden_params,
        flops=hidden_params,
        entrada_excel=prev_out_repr,
        salida_excel=hidden_out_repr,
        total_parametros=hidden_params,
        is_blue=False,
        formula=f"({flat_tot} × {hid}) + {hid} = {fmt_int(hidden_params)} parámetros",
        formula_excel=f"=({hid}*{flat_tot})+{hid}",
        note=f"{hid} neuronas ocultas que combinan los patrones de los mapas convolucionales.",
        color="#fbbf24",
    ))
    prev_out_repr = hidden_out_repr

    # 5. Capa de salida (proyección a los logits de las tres clases)
    sm_params = n_out * hid + n_out
    logits_out_repr = f"{n_out} logits  ({n_out})"
    rows.append(ArchRow(
        stage="Salida",
        capa="Salida",
        operation="Capa de salida (proyección lineal a los logits)",
        input_shape=(hid,),
        output_shape=(n_out,),
        elements=n_out,
        params=sm_params,
        flops=sm_params,
        entrada_excel=prev_out_repr,
        salida_excel=logits_out_repr,
        total_parametros=sm_params,
        is_blue=False,
        formula=f"({hid} × {n_out}) + {n_out} = {fmt_int(sm_params)} parámetros",
        formula_excel=f"=({n_out}*{hid})+{n_out}",
        note="Convierte las activaciones ocultas en una puntuación por clase.",
        color="#34d399",
    ))
    prev_out_repr = logits_out_repr

    # 6. Softmax: convierte los logits en probabilidades
    sm_out_repr = f"{n_out} clases (Trucha, Mojarra, Pavón)"
    rows.append(ArchRow(
        stage="Softmax",
        capa="Softmax",
        operation="Softmax: normaliza los logits en probabilidades",
        input_shape=(n_out,),
        output_shape=(n_out,),
        elements=n_out,
        params=0,
        flops=n_out,
        entrada_excel=prev_out_repr,
        salida_excel=sm_out_repr,
        total_parametros=0,
        is_blue=False,
        formula="pᵢ = exp(zᵢ − max z) / Σ exp(zⱼ − max z)",
        formula_excel="=SUMA(EXP(z-max))/...",
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
# Laboratorio numérico: pasos reales de la CNN
# ---------------------------------------------------------------------------
@dataclass
class LabStep:
    """Un fotograma del recorrido numérico de una capa real de la CNN."""

    index: int
    title: str
    detail: str
    formula: str
    source: np.ndarray
    source_highlight: tuple[int, int, int, int] | None
    source_label: str
    kernel: np.ndarray | None
    kernel_label: str
    target: np.ndarray
    target_highlight: tuple[int, int, int, int] | None
    target_label: str
    phase: str = ""


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        side = int(np.ceil(np.sqrt(arr.size)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded.reshape(side, side)
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba una matriz 2D; se recibió {arr.shape}")
    return arr


def conv1_frames(
    input_2d: Any,
    kernel: Any,
    bias: float,
    crop: int = 8,
) -> list[LabStep]:
    """Convolución real de un filtro aprendido sobre un recorte de la entrada.

    El recorte central se elige de forma que cada ventana 3×3 esté completa
    dentro de la imagen. Así, el mapa calculado coincide exactamente con la
    activación que produce la capa Conv 1 del modelo sobre esa región.
    """
    image = _as_2d(input_2d)
    k = np.asarray(kernel, dtype=np.float32)
    if k.shape != (3, 3):
        k = np.resize(k, (3, 3)).astype(np.float32)
    h, w = image.shape
    side = max(2, min(int(crop), h - 2, w - 2))
    i0 = max(1, (h - side) // 2)
    j0 = max(1, (w - side) // 2)
    i0 = min(i0, h - side - 1)
    j0 = min(j0, w - side - 1)
    source = image[i0 - 1: i0 + side + 1, j0 - 1: j0 + side + 1]
    raw = np.zeros((side, side), dtype=np.float32)
    frames: list[LabStep] = []
    bias = float(bias)
    total = side * side
    for a in range(side):
        for b in range(side):
            window = source[a:a + 3, b:b + 3]
            products = window * k
            z = float(products.sum()) + bias
            raw[a, b] = z
            terms = " + ".join(
                f"({window[u, v]:.2f}·{k[u, v]:+.2f})" for u in range(3) for v in range(3)
            )
            frames.append(LabStep(
                index=len(frames),
                phase="conv",
                title=f"Conv 1 · celda {a * side + b + 1}/{total}",
                detail=(
                    f"La ventana 3×3 se coloca en la fila {i0 + a}, columna {j0 + b} de la "
                    f"imagen de {h}×{w} ya preprocesada. Se multiplica por el filtro aprendido, "
                    f"se suman los 9 productos y se añade el sesgo b = {bias:+.4f}. "
                    f"El resultado alimenta la posición ({a}, {b}) del mapa."
                ),
                formula=f"{terms} + b = {z:+.4f}",
                source=source.copy(),
                source_highlight=(a, b, 3, 3),
                source_label=f"Entrada real · ventana {source.shape[0]}×{source.shape[1]}",
                kernel=k.copy(),
                kernel_label="Filtro aprendido 3×3",
                target=raw.copy(),
                target_highlight=(a, b, 1, 1),
                target_label=f"Mapa Conv 1 · {side}×{side}",
            ))
    activated = np.maximum(0.0, raw)
    negatives = int((raw < 0).sum())
    frames.append(LabStep(
        index=len(frames),
        phase="relu",
        title="ReLU sobre el mapa convolucional",
        detail=(
            f"Se aplica max(0, z) a las {raw.size} celdas del mapa. "
            f"{negatives} valores negativos pasan a 0 y los positivos se conservan. "
            f"Este mapa activado es el que entra al Max Pooling de la red."
        ),
        formula="a = max(0, z)",
        source=raw.copy(),
        source_highlight=None,
        source_label="Mapa Conv 1 (z)",
        kernel=None,
        kernel_label="",
        target=activated.copy(),
        target_highlight=None,
        target_label="Mapa activado (ReLU)",
    ))
    return frames


def hidden_frames(
    flat_input: Any,
    weights: Any,
    bias: float,
    neuron: int,
    top_k: int = 64,
) -> list[LabStep]:
    """Producto punto de una neurona oculta con sus entradas reales.

    La capa oculta recibe el vector aplanado y calcula z = Σ x·w + b. Para poder
    leer los números se muestran las ``top_k`` entradas con mayor contribución
    absoluta; el resto se resume en un término de "resto".
    """
    x = np.asarray(flat_input, dtype=np.float32).ravel()
    w = np.asarray(weights, dtype=np.float32).ravel()
    if x.size != w.size:
        raise ValueError(f"Entradas ({x.size}) y pesos ({w.size}) no coinciden")
    n = x.size
    k = max(1, min(int(top_k), n))
    side = int(np.ceil(np.sqrt(k)))
    contributions = x * w
    order = np.argsort(-np.abs(contributions))[:k]
    src = np.zeros(side * side, dtype=np.float32)
    wet = np.zeros(side * side, dtype=np.float32)
    prod = np.zeros(side * side, dtype=np.float32)
    src[:k] = x[order]
    wet[:k] = w[order]
    prod[:k] = contributions[order]
    src = src.reshape(side, side)
    wet = wet.reshape(side, side)
    prod = prod.reshape(side, side)
    top_sum = float(prod.sum())
    total_sum = float(contributions.sum())
    residual = total_sum - top_sum
    z = total_sum + float(bias)
    activation = max(0.0, z)
    frames = [
        LabStep(
            index=0,
            phase="products",
            title=f"Capa oculta · neurona {neuron + 1} · productos x·w",
            detail=(
                f"La neurona {neuron + 1} recibe {n} activaciones del vector aplanado. "
                f"Se muestran las {k} entradas con mayor |x·w|. Cada celda es el producto "
                f"de una entrada por su peso; los pesos pertenecen al modelo cargado."
            ),
            formula=f"x·w → {k} productos (el resto se suma aparte)",
            source=src,
            source_highlight=None,
            source_label=f"Entradas x · {k} de {n}",
            kernel=wet,
            kernel_label=f"Pesos w de la neurona {neuron + 1}",
            target=prod,
            target_highlight=None,
            target_label="Productos x·w",
        ),
        LabStep(
            index=1,
            phase="sum",
            title="Suma de productos + sesgo → z",
            detail=(
                f"Σ de las {k} entradas mostradas = {top_sum:+.4f}. "
                f"Resto de las {n - k} entradas = {residual:+.4f}. "
                f"Sesgo b = {float(bias):+.4f}. Por tanto, z = {z:+.4f}. "
                f"Este z es la salida lineal de la neurona antes de la activación."
            ),
            formula=f"Σ_top ({top_sum:+.4f}) + resto ({residual:+.4f}) + b ({float(bias):+.4f}) = {z:+.4f}",
            source=prod,
            source_highlight=None,
            source_label="Productos x·w",
            kernel=None,
            kernel_label="",
            target=np.array([[z]], dtype=np.float32),
            target_highlight=None,
            target_label="z de la neurona",
        ),
        LabStep(
            index=2,
            phase="relu",
            title="ReLU → activación de la neurona",
            detail=(
                f"La capa oculta aplica max(0, z). Con z = {z:+.4f}, la activación final "
                f"de la neurona es {activation:+.4f}. Esta activación viaja a la capa de salida."
            ),
            formula=f"a = max(0, {z:+.4f}) = {activation:+.4f}",
            source=np.array([[z]], dtype=np.float32),
            source_highlight=None,
            source_label="z (salida lineal)",
            kernel=None,
            kernel_label="",
            target=np.array([[activation]], dtype=np.float32),
            target_highlight=None,
            target_label="Activación ReLU",
        ),
    ]
    return frames


def softmax_frames(
    logits: Any,
    labels: list[str] | tuple[str, ...],
    true_label: int | None = None,
) -> list[LabStep]:
    """Paso a paso real de la capa de salida Softmax."""
    z = np.asarray(logits, dtype=np.float32).ravel()
    if z.size < 2:
        raise ValueError("Softmax necesita al menos dos clases")
    names = [str(v) for v in labels][: z.size]
    if len(names) < z.size:
        names += [f"Clase {i + 1}" for i in range(len(names), z.size)]
    shifted = z - float(z.max())
    exp = np.exp(shifted)
    total = float(exp.sum())
    probs = exp / max(total, 1e-12)

    z_row = z.reshape(1, -1)
    e_row = exp.reshape(1, -1)
    p_row = probs.reshape(1, -1)
    detail_values = ", ".join(
        f"{names[i]} z={z[i]:+.4f} → e={exp[i]:.4f} → p={probs[i]:.4f}"
        for i in range(z.size)
    )
    predicted = int(np.argmax(probs))
    frames = [
        LabStep(
            index=0,
            phase="exp",
            title="Salida · logits → exponenciales",
            detail=(
                f"La capa de salida produce los logits {np.array2string(z, precision=4)}. "
                f"Se resta el máximo ({float(z.max()):+.4f}) para evitar desbordamientos y "
                f"se aplica exp. {detail_values}."
            ),
            formula=f"eᵢ = exp(zᵢ − max z) = {np.array2string(exp, precision=4)}",
            source=z_row.copy(),
            source_highlight=None,
            source_label="Logits z (capa de salida)",
            kernel=None,
            kernel_label="",
            target=e_row.copy(),
            target_highlight=None,
            target_label="Exponenciales",
        ),
        LabStep(
            index=1,
            phase="normalize",
            title="Normalización de probabilidades",
            detail=(
                f"La suma de exponenciales es {total:.4f}. Al dividir cada una entre la suma, "
                f"se obtiene una distribución que suma 1. La clase predicha es "
                f"«{names[predicted]}» con {probs[predicted] * 100:.2f} % de confianza."
            ),
            formula="pᵢ = eᵢ / Σe = " + np.array2string(probs, precision=4),
            source=e_row.copy(),
            source_highlight=None,
            source_label="Exponenciales",
            kernel=None,
            kernel_label="",
            target=p_row.copy(),
            target_highlight=(0, predicted, 1, 1),
            target_label="Probabilidades Softmax",
        ),
        LabStep(
            index=2,
            phase="done",
            title="Predicción final de la red",
            detail=(
                f"La red asigna {', '.join(f'{names[i]} {probs[i] * 100:.2f}%' for i in range(z.size))}. "
                + (
                    f"La etiqueta real de la imagen es «{names[true_label]}»."
                    if true_label is not None and 0 <= int(true_label) < z.size
                    else "No hay etiqueta real disponible para esta muestra."
                )
            ),
            formula=f"argmax(p) = {predicted} → «{names[predicted]}» ({probs[predicted] * 100:.2f} %)",
            source=p_row.copy(),
            source_highlight=None,
            source_label="Probabilidades",
            kernel=None,
            kernel_label="",
            target=p_row.copy(),
            target_highlight=(0, predicted, 1, 1),
            target_label="Clase predicha",
        ),
    ]
    return frames


# ---------------------------------------------------------------------------
# Descripción textual usada por la auditoría y la interfaz
# ---------------------------------------------------------------------------
MANUAL_ENGINE_CHECKS = (
    ("Lectura de imágenes", "OpenCV solo decodifica píxeles; no aprende nada."),
    ("Convolución", "core/image_ops.convolve2d: ventanas + producto punto en NumPy."),
    ("Forward CNN", "core/cnn.py: Conv2D, ReLU, MaxPool2D y Flatten propios."),
    ("Backpropagation", "core/cnn.py: gradientes analíticos de conv, pooling y capas ocultas."),
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
