"""Descarga un dataset de ejemplo de peces desde Wikimedia Commons.

Uso:
    python tools/download_dataset.py --per-class 35

Las imagenes quedan en ``Dataset Imagenes/<clase>/`` y se genera un archivo
``CREDITOS.txt`` con autor y licencia de cada imagen.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Dataset Imagenes"
API = "https://commons.wikimedia.org/w/api.php"
UA = "Lab6FishClassifier/1.0 (educational project; contact: student@example.com)"

CLASSES = {
    "trucha": [
        "Category:Oncorhynchus mykiss",
    ],
    "mojarra": [
        "Category:Oreochromis niloticus",
        "Category:Tilapia",
    ],
    "pavon": [
        "Category:Cichla ocellaris",
        "Category:Cichla temensis",
        "Category:Cichla",
    ],
}

SEARCH_QUERIES = {
    "trucha": ["rainbow trout fish", "Oncorhynchus mykiss fish"],
    "mojarra": ["Oreochromis niloticus fish", "Nile tilapia fish"],
    "pavon": ["Cichla ocellaris fish", "peacock bass fish", "Cichla temensis"],
}

VALID_EXT = {".jpg", ".jpeg", ".png"}
SKIP_WORDS = {
    "map", "range", "distribution", "logo", "icon", "drawing", "illustration",
    "skeleton", "skull", "bone", "egg", "larva", "fry", "scale", "otolith",
    "diagram", "anatomy", "xray", "x-ray", "fossil", "stamp", "coin",
    "sign", "park", "hatchery", "fishing", "angler", "cooked", "dish",
    "aquarium_tank", "pond", "river", "lake", "logo", "seal", "poster",
}


def api_get(params: dict) -> dict:
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def category_files(category: str, limit: int = 200) -> list[dict]:
    data = api_get({
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "900",
    })
    return data.get("query", {}).get("pages", [])


def search_files(query: str, limit: int = 60) -> list[dict]:
    data = api_get({
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "900",
    })
    return data.get("query", {}).get("pages", [])


def clean_name(name: str) -> str:
    name = re.sub(r"^File:", "", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:80]


def looks_like_fish_photo(title: str) -> bool:
    low = title.lower()
    if any(w in low for w in SKIP_WORDS):
        return False
    return True


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            blob = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! error {exc}")
        return False
    if len(blob) < 8000:
        return False
    head = blob[:12]
    if not (head.startswith(b"\xff\xd8") or head.startswith(b"\x89PNG")):
        return False
    dest.write_bytes(blob)
    return True


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=35)
    args = ap.parse_args()

    credits: list[str] = ["Dataset de ejemplo - Wikimedia Commons", ""]
    total = 0
    for cls, categories in CLASSES.items():
        out = DATASET / cls
        out.mkdir(parents=True, exist_ok=True)
        have = len([p for p in out.iterdir() if p.suffix.lower() in VALID_EXT])
        if have >= args.per_class:
            print(f"[{cls}] ya tiene {have} imagenes, se omite")
            total += have
            continue
        seen: set[str] = set()
        state = {"count": have}
        print(f"[{cls}] descargando hasta {args.per_class} imagenes...")

        def consume(pages: list[dict]) -> None:
            for page in pages:
                if state["count"] >= args.per_class:
                    return
                title = page.get("title", "")
                if not looks_like_fish_photo(title):
                    continue
                info = (page.get("imageinfo") or [{}])[0]
                mime = info.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue
                if info.get("width", 0) < 500 or info.get("height", 0) < 300:
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                ext = ".jpg" if mime == "image/jpeg" else ".png"
                n = state["count"]
                dest = out / f"{cls}_{n + 1:03d}_{clean_name(title)}{ext}"
                print(f"    + {title}")
                ok = download(url, dest)
                if not ok and info.get("url") and info.get("url") != url:
                    ok = download(info["url"], dest)
                if ok:
                    meta = info.get("extmetadata") or {}
                    author = strip_html((meta.get("Artist") or {}).get("value", "Desconocido"))
                    lic = strip_html((meta.get("LicenseShortName") or {}).get("value", "?"))
                    credits.append(f"{cls}/{dest.name}\t{author}\t{lic}\t{title}")
                    state["count"] += 1
                time.sleep(0.15)

        for cat in categories:
            if state["count"] >= args.per_class:
                break
            try:
                pages = category_files(cat)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! categoria {cat}: {exc}")
                continue
            print(f"  - {cat}: {len(pages)} archivos")
            consume(pages)

        # Relleno por busqueda libre si la categoria no alcanzo.
        for query in SEARCH_QUERIES.get(cls, []):
            if state["count"] >= args.per_class:
                break
            try:
                pages = search_files(query)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! busqueda {query}: {exc}")
                continue
            print(f"  - busqueda '{query}': {len(pages)} archivos")
            consume(pages)

        total += state["count"]
        print(f"[{cls}] total {state['count']}")

    (DATASET / "CREDITOS.txt").write_text("\n".join(credits), encoding="utf-8")
    print(f"\nListo: {total} imagenes en {DATASET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
