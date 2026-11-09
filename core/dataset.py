"""Gestión del dataset de peces: escaneo, división, importación y estructura."""
from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.config import CLASSES, CLASS_LABELS, IMAGE_EXTENSIONS, MAX_DATASET_IMAGES


@dataclass
class Sample:
    path: Path
    label: int
    class_name: str

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class DatasetInfo:
    root: Path
    samples: list[Sample] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    invalid: list[str] = field(default_factory=list)
    found: int = 0
    limit: int = MAX_DATASET_IMAGES
    truncated: bool = False

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    @property
    def class_count(self) -> int:
        return sum(1 for c in CLASSES if self.counts.get(c, 0) > 0)

    @property
    def balance(self) -> float:
        """1.0 = clases perfectamente balanceadas."""
        vals = [self.counts.get(c, 0) for c in CLASSES if self.counts.get(c, 0) > 0]
        if len(vals) <= 1:
            return 0.0
        return min(vals) / max(vals)

    def by_class(self) -> dict[str, list[Sample]]:
        out = {c: [] for c in CLASSES}
        for s in self.samples:
            out[s.class_name].append(s)
        return out

    def labels(self) -> list[str]:
        return [CLASS_LABELS[c] for c in CLASSES]


def ensure_structure(root: str | Path) -> Path:
    root = Path(root)
    for cls in CLASSES:
        (root / cls).mkdir(parents=True, exist_ok=True)
    return root


def scan_dataset(root: str | Path, limit: int = MAX_DATASET_IMAGES) -> DatasetInfo:
    """Escanea ``root/clase/*`` aceptando cualquier cantidad de imágenes.

    Si se supera ``limit`` se conserva una muestra equilibrada por clase
    (reparto round-robin) y se marca ``truncated`` para avisar en la interfaz.
    """
    root = Path(root)
    info = DatasetInfo(root=root, counts={c: 0 for c in CLASSES}, limit=int(limit))
    if not root.exists():
        return info

    per_class: dict[str, list[Path]] = {}
    for cls in CLASSES:
        folder = root / cls
        files: list[Path] = []
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if p.name.startswith("."):
                    continue
                try:
                    if p.stat().st_size < 64:
                        info.invalid.append(str(p))
                        continue
                except OSError:
                    info.invalid.append(str(p))
                    continue
                files.append(p)
        per_class[cls] = files
        info.found += len(files)

    total = info.found
    if total <= limit:
        chosen = per_class
    else:
        info.truncated = True
        chosen = {c: [] for c in CLASSES}
        indexes = {c: 0 for c in CLASSES}
        taken = 0
        while taken < limit:
            progressed = False
            for cls in CLASSES:
                if taken >= limit:
                    break
                idx = indexes[cls]
                if idx < len(per_class[cls]):
                    chosen[cls].append(per_class[cls][idx])
                    indexes[cls] = idx + 1
                    taken += 1
                    progressed = True
            if not progressed:
                break

    for label, cls in enumerate(CLASSES):
        for p in chosen.get(cls, []):
            info.samples.append(Sample(path=p, label=label, class_name=cls))
            info.counts[cls] += 1
    return info


def _allocate_split(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Reparte ``n`` elementos respetando las proporciones sin dejar clases vacías.

    Garantiza al menos 1 elemento en cada conjunto con proporción > 0 cuando
    hay suficientes muestras; el resto se reparte por mayor resto.
    """
    positive = [i for i, r in enumerate(ratios) if r > 0]
    alloc = [0, 0, 0]
    if not positive or n <= 0:
        alloc[0] = n
        return tuple(alloc)  # type: ignore[return-value]
    if n < len(positive):
        for i in sorted(positive, key=lambda k: ratios[k], reverse=True)[:n]:
            alloc[i] = 1
        return tuple(alloc)  # type: ignore[return-value]

    remaining = n
    for i in positive:
        alloc[i] = 1
        remaining -= 1
    if remaining > 0:
        quotas = [remaining * (ratios[i] if i in positive else 0.0) for i in range(3)]
        base = [int(q) for q in quotas]
        for i in range(3):
            alloc[i] += base[i]
        rest = remaining - sum(base)
        order = sorted(range(3), key=lambda i: (quotas[i] - base[i], ratios[i]), reverse=True)
        for i in order[:rest]:
            alloc[i] += 1
    return tuple(alloc)  # type: ignore[return-value]


def split_dataset(
    samples: list[Sample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratified: bool = True,
) -> dict[str, list[Sample]]:
    """Divide el dataset en train/val/test de forma estratificada."""
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Las proporciones deben sumar más de cero")
    train_ratio, val_ratio, test_ratio = (r / total for r in (train_ratio, val_ratio, test_ratio))
    rng = random.Random(seed)
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    ratios = (train_ratio, val_ratio, test_ratio)

    groups: dict[int, list[Sample]] = {}
    if stratified:
        for s in samples:
            groups.setdefault(s.label, []).append(s)
    else:
        groups[0] = list(samples)

    for group in groups.values():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue
        counts = _allocate_split(n, ratios)
        a, b = counts[0], counts[0] + counts[1]
        splits["train"].extend(group[:a])
        splits["val"].extend(group[a:b])
        splits["test"].extend(group[b:])

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def unique_destination(folder: Path, filename: str) -> Path:
    stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
    dest = folder / f"{stem}{suffix}"
    i = 1
    while dest.exists():
        dest = folder / f"{stem}_{i}{suffix}"
        i += 1
    return dest


def import_images(
    paths: list[str | Path],
    class_name: str,
    root: str | Path,
    max_total: int | None = None,
) -> list[Path]:
    """Copia imágenes a la carpeta de la clase indicada respetando el tope global."""
    if class_name not in CLASSES:
        raise ValueError(f"Clase no válida: {class_name}")
    folder = Path(root) / class_name
    folder.mkdir(parents=True, exist_ok=True)
    room: int | None = None
    if max_total is not None:
        existing = 0
        for cls in CLASSES:
            sub = Path(root) / cls
            if sub.exists():
                existing += sum(
                    1 for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
        room = max(0, int(max_total) - existing)
    copied: list[Path] = []
    for raw in paths:
        if room is not None and len(copied) >= room:
            break
        src = Path(raw)
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            if src.resolve() == folder.resolve() / src.name:
                continue  # la imagen ya está en su carpeta de destino
            dest = unique_destination(folder, src.name)
            shutil.copy2(src, dest)
            copied.append(dest)
        except (OSError, shutil.Error):
            continue  # un archivo bloqueado no debe cortar el resto del lote
    return copied


def delete_samples(paths: list[str | Path]) -> int:
    removed = 0
    for raw in paths:
        p = Path(raw)
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def file_digest(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_summary(info: DatasetInfo) -> str:
    if info.is_empty:
        return "Dataset vacío"
    parts = [f"{CLASS_LABELS[c]}: {info.counts.get(c, 0)}" for c in CLASSES]
    return f"{info.total} imágenes · " + " · ".join(parts)
