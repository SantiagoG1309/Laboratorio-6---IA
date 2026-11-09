"""Amplía el dataset hasta un mínimo por clase usando iNaturalist y Wikimedia.

Uso:
    python tools/dataset/expand_dataset.py --target 120

- iNaturalist: observaciones de grado investigación con licencia CC.
- Wikimedia Commons: categorías y búsquedas como respaldo.
- Valida que cada archivo sea una imagen legible y descarta duplicados
  (MD5 + hash perceptual) contra todo el dataset.
- Anexa las atribuciones a CREDITOS.txt / CREDITOS_INATURALIST.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"
UA = "Lab6FishClassifier/1.0 (educational project)"

INAT_API = "https://api.inaturalist.org/v1/observations"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"

TAXA = {
    "trucha": ["Oncorhynchus mykiss"],
    "mojarra": ["Oreochromis niloticus", "Oreochromis mossambicus"],
    "pavon": ["Cichla ocellaris", "Cichla temensis", "Cichla orinocensis", "Cichla monoculus"],
}

COMMONS_CATEGORIES = {
    "trucha": ["Category:Oncorhynchus mykiss", "Category:Rainbow trout"],
    "mojarra": ["Category:Oreochromis niloticus", "Category:Tilapia"],
    "pavon": ["Category:Cichla ocellaris", "Category:Cichla temensis", "Category:Cichla"],
}

COMMONS_QUERIES = {
    "trucha": ["rainbow trout fish", "Oncorhynchus mykiss"],
    "mojarra": ["Oreochromis niloticus", "Nile tilapia"],
    "pavon": ["Cichla ocellaris", "peacock bass", "Cichla temensis"],
}

SKIP_WORDS = {
    "map", "range", "distribution", "logo", "icon", "drawing", "illustration",
    "skeleton", "skull", "bone", "egg", "larva", "fry", "scale", "otolith",
    "diagram", "anatomy", "xray", "x-ray", "fossil", "stamp", "coin",
    "sign", "park", "hatchery", "fishing", "angler", "cooked", "dish",
    "recipe", "aquarium_tank", "pond", "river", "lake", "seal", "poster",
}
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


# ---------------------------------------------------------------------------
# Utilidades de red
# ---------------------------------------------------------------------------
def _request(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url: str, params: dict, tries: int = 3) -> dict:
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            return json.loads(_request(full).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"    ! reintento ({exc})")
            time.sleep(2 + attempt * 3)
    return {}


def download(url: str, tries: int = 3) -> bytes | None:
    for attempt in range(tries):
        try:
            blob = _request(url, timeout=90)
            if len(blob) < 8000:
                return None
            if not (blob[:2] == b"\xff\xd8" or blob[:4] == b"\x89PNG" or blob[:4] == b"RIFF"):
                return None
            return blob
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return None
            time.sleep(1.5 + attempt * 2)
        except Exception:  # noqa: BLE001
            time.sleep(1.5 + attempt * 2)
    return None


# ---------------------------------------------------------------------------
# Hashes / validación
# ---------------------------------------------------------------------------
def md5(blob: bytes) -> str:
    return hashlib.md5(blob).hexdigest()


def dhash(img: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def ahash(img: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    value = 0
    for bit in (resized > mean).flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class Deduper:
    def __init__(self) -> None:
        self.md5s: set[str] = set()
        self.hashes: list[tuple[int, int]] = []

    def load_existing(self) -> None:
        for cls in TAXA:
            folder = DS / cls
            if not folder.exists():
                continue
            for p in folder.iterdir():
                if not p.is_file() or p.suffix.lower() not in VALID_EXT:
                    continue
                blob = p.read_bytes()
                self.md5s.add(md5(blob))
                img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.hashes.append((dhash(img), ahash(img)))

    def is_duplicate(self, blob: bytes, img: np.ndarray) -> bool:
        if md5(blob) in self.md5s:
            return True
        dh, ah = dhash(img), ahash(img)
        for d, a in self.hashes:
            if hamming(dh, d) <= 6 and hamming(ah, a) <= 8:
                return True
        return False

    def add(self, blob: bytes, img: np.ndarray) -> None:
        self.md5s.add(md5(blob))
        self.hashes.append((dhash(img), ahash(img)))


def validate(blob: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if min(h, w) < 220 or h * w < 90_000:
        return None
    ratio = w / max(1, h)
    if not (0.42 <= ratio <= 2.6):
        return None
    # Imágenes casi uniformes (logos, fondos) fuera.
    if float(img.std()) < 12.0:
        return None
    return img


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def clean(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:limit].strip("_")


def next_index(folder: Path) -> int:
    highest = 0
    for p in folder.iterdir() if folder.exists() else []:
        m = re.match(r"^[a-z]+_(\d+)", p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def used_observations(folder: Path) -> set[str]:
    """IDs de observaciones de iNaturalist ya presentes (una foto por obs)."""
    used: set[str] = set()
    for p in folder.iterdir() if folder.exists() else []:
        m = re.search(r"_inat_[A-Za-z_]+_(\d+)(?:_|\.)", p.name)
        if m:
            used.add(m.group(1))
    return used


def save(
    cls: str,
    blob: bytes,
    img: np.ndarray,
    name: str,
    deduper: Deduper,
    credits: list[str],
    line: str,
) -> bool:
    folder = DS / cls
    folder.mkdir(parents=True, exist_ok=True)
    if deduper.is_duplicate(blob, img):
        return False
    dest = folder / name
    dest.write_bytes(blob)
    deduper.add(blob, img)
    credits.append(line)
    return True


def from_inaturalist(
    cls: str,
    taxa: list[str],
    target: int,
    deduper: Deduper,
    credits: list[str],
    per_page: int = 100,
) -> int:
    folder = DS / cls
    count = len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])
    used = used_observations(folder)
    print(f"[{cls}] iNaturalist · inicio {count} / {target} · {len(used)} obs ya usadas")
    for taxon in taxa:
        if count >= target:
            break
        page = 1
        while count < target and page <= 12:
            params = {
                "taxon_name": taxon,
                "photos": "true",
                "quality_grade": "research",
                "photo_license": LICENSES,
                "per_page": str(per_page),
                "page": str(page),
                "order_by": "votes",
                "order": "desc",
            }
            try:
                data = get_json(INAT_API, params)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {taxon}: {exc}")
                break
            results = data.get("results", [])
            if not results:
                break
            print(f"  - {taxon} p{page}: {len(results)} obs")
            for obs in results:
                if count >= target:
                    break
                obs_id = str(obs.get("id", ""))
                # Una sola foto por observación: evita que el mismo pez quede
                # repartido entre train y test (fuga de datos).
                if not obs_id or obs_id in used:
                    continue
                photo = None
                for candidate in obs.get("photos", []):
                    if candidate.get("license_code") not in LICENSES.split(","):
                        continue
                    dims = candidate.get("original_dimensions") or {}
                    if min(dims.get("width", 0), dims.get("height", 0)) < 400:
                        continue
                    if candidate.get("url"):
                        photo = candidate
                        break
                if photo is None:
                    continue
                base = photo.get("url", "")
                blob = None
                for variant in ("large", "medium"):
                    candidate_url = re.sub(r"/(square|thumb|small|medium|large)\.", f"/{variant}.", base)
                    blob = download(candidate_url)
                    if blob:
                        break
                if not blob:
                    continue
                img = validate(blob)
                if img is None:
                    continue
                idx = next_index(folder)
                photo_id = photo.get("id", "0")
                name = f"{cls}_{idx:03d}_inat_{clean(taxon)}_{obs_id}_{photo_id}.jpg"
                ok = save(
                    cls, blob, img, name, deduper, credits,
                    f"{cls}/{name}\t{obs.get('user', {}).get('login', '?')}\t"
                    f"{photo.get('license_code', '?')}\t"
                    f"https://www.inaturalist.org/observations/{obs_id}",
                )
                if ok:
                    used.add(obs_id)
                    count += 1
                    print(f"    + {name} ({count}/{target})")
                time.sleep(0.05)
            page += 1
            if data.get("total_results", 0) and page * per_page >= int(data["total_results"]):
                break
    print(f"[{cls}] iNaturalist · total {count}")
    return count


def _commons_pages(params: dict) -> list[dict]:
    try:
        data = get_json(COMMONS_API, params)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! commons: {exc}")
        return []
    return data.get("query", {}).get("pages", [])


def commons_pages(category: str, limit: int = 200) -> list[dict]:
    return _commons_pages({
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "1000",
    })


def commons_search(query: str, limit: int = 100) -> list[dict]:
    return _commons_pages({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "1000",
    })


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def from_commons(
    cls: str,
    target: int,
    deduper: Deduper,
    credits: list[str],
) -> int:
    folder = DS / cls
    count = len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])
    print(f"[{cls}] Wikimedia · inicio {count} / {target}")
    seen_urls: set[str] = set()

    def consume(pages: list[dict]) -> None:
        nonlocal count
        for page in pages:
            if count >= target:
                return
            title = page.get("title", "")
            low = title.lower()
            if any(w in low for w in SKIP_WORDS):
                continue
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if info.get("width", 0) < 500 or info.get("height", 0) < 350:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            blob = download(url)
            if not blob and info.get("url") and info["url"] != url:
                blob = download(info["url"])
            if not blob:
                continue
            img = validate(blob)
            if img is None:
                continue
            idx = next_index(folder)
            name = f"{cls}_{idx:03d}_{clean(title)[:50]}.jpg"
            meta = info.get("extmetadata") or {}
            author = strip_html((meta.get("Artist") or {}).get("value", "Desconocido"))
            lic = strip_html((meta.get("LicenseShortName") or {}).get("value", "?"))
            ok = save(
                cls, blob, img, name, deduper, credits,
                f"{cls}/{name}\t{author}\t{lic}\t{title}",
            )
            if ok:
                count += 1
                print(f"    + {name} ({count}/{target})")
            time.sleep(0.12)

    for category in COMMONS_CATEGORIES.get(cls, []):
        if count >= target:
            break
        pages = commons_pages(category)
        print(f"  - {category}: {len(pages)} archivos")
        consume(pages)
    for query in COMMONS_QUERIES.get(cls, []):
        if count >= target:
            break
        pages = commons_search(query)
        print(f"  - búsqueda '{query}': {len(pages)} archivos")
        consume(pages)
    print(f"[{cls}] Wikimedia · total {count}")
    return count


def append_credits(path: Path, header: str, lines: list[str]) -> None:
    if not lines:
        return
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"
    else:
        existing = header + "\n\n"
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--class", dest="only", default=None)
    args = ap.parse_args()

    deduper = Deduper()
    deduper.load_existing()
    print(f"hashes existentes: {len(deduper.hashes)} imágenes")

    inat_credits: list[str] = []
    commons_credits: list[str] = []
    classes = [args.only] if args.only else list(TAXA)
    for cls in classes:
        folder = DS / cls
        folder.mkdir(parents=True, exist_ok=True)
        have = len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])
        if have >= args.target:
            print(f"[{cls}] ya tiene {have}, se omite")
            continue
        count = from_inaturalist(cls, TAXA[cls], args.target, deduper, inat_credits)
        if count < args.target:
            from_commons(cls, args.target, deduper, commons_credits)
        # Guardado incremental: si el proceso se interrumpe, las atribuciones
        # de lo ya descargado no se pierden.
        append_credits(
            DS / "CREDITOS_INATURALIST.txt",
            "Fotografías adicionales: iNaturalist (www.inaturalist.org)",
            inat_credits,
        )
        append_credits(DS / "CREDITOS.txt", "Dataset de ejemplo - Wikimedia Commons", commons_credits)
        inat_credits.clear()
        commons_credits.clear()

    append_credits(
        DS / "CREDITOS_INATURALIST.txt",
        "Fotografías adicionales: iNaturalist (www.inaturalist.org)",
        inat_credits,
    )
    append_credits(DS / "CREDITOS.txt", "Dataset de ejemplo - Wikimedia Commons", commons_credits)

    print("\nResumen final:")
    for cls in TAXA:
        folder = DS / cls
        n = len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])
        print(f"  {cls}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
