"""Rellena el dataset hasta un objetivo usando categorias concretas y filtros estrictos."""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_dataset as dd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"

EXTRA_SKIP = {
    "bird", "heron", "eagle", "kingfisher", "taxidermy", "victorian", "canal",
    "landscape", "market", "food", "dish", "cooked", "fried", "steamed",
    "plate", "restaurant", "recipe", "soup", "grill", "bbq", "smoked",
    "fisherman", "angler", "fishing", "boat", "zoo", "sign", "poster",
    "drawing", "print", "lithograph", "engraving", "specimen", "museum",
    "skeleton", "skull", "egg", "nest", "juvenile", "fry", "larva",
}

CONFIG = {
    "trucha": {
        "target": 34,
        "cats": [
            "Category:Oncorhynchus mykiss",
            "Category:Rainbow trout",
            "Category:Trout",
        ],
        "queries": ["rainbow trout underwater", "trout in river"],
    },
    "mojarra": {
        "target": 34,
        "cats": [
            "Category:Oreochromis niloticus",
            "Category:Oreochromis mossambicus",
            "Category:Oreochromis",
            "Category:Tilapia",
        ],
        "queries": ["tilapia fish underwater", "Oreochromis fish"],
    },
    "pavon": {
        "target": 34,
        "cats": [
            "Category:Cichla ocellaris",
            "Category:Cichla temensis",
            "Category:Cichla",
        ],
        "queries": ["peacock bass underwater", "Cichla fish"],
    },
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def title_key(name: str) -> str:
    # quita prefijo clase_numero_ y extension
    m = re.match(r"^[a-z]+_\d+_(.*)$", name, re.IGNORECASE)
    return (m.group(1) if m else name).rsplit(".", 1)[0].lower()


def acceptable(title: str) -> bool:
    low = title.lower()
    return not any(w in low for w in EXTRA_SKIP)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    for cls, cfg in CONFIG.items():
        out = DS / cls
        out.mkdir(parents=True, exist_ok=True)
        files = [p for p in out.iterdir() if p.suffix.lower() in dd.VALID_EXT]
        hashes = {md5(p) for p in files}
        keys = {title_key(p.name) for p in files}
        state = {"count": len(files)}
        print(f"[{cls}] inicio {state['count']} / {cfg['target']}")

        def consume(pages: list[dict]) -> None:
            for page in pages:
                if state["count"] >= cfg["target"]:
                    return
                title = page.get("title", "")
                if not acceptable(title):
                    continue
                info = (page.get("imageinfo") or [{}])[0]
                mime = info.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue
                if info.get("width", 0) < 500 or info.get("height", 0) < 300:
                    continue
                key = title_key(dd.clean_name(title))
                if key in keys:
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url:
                    continue
                tmp = out / "_tmp_download"
                if not dd.download(url, tmp):
                    continue
                h = md5(tmp)
                if h in hashes:
                    tmp.unlink(missing_ok=True)
                    continue
                hashes.add(h)
                keys.add(key)
                n = state["count"] + 1
                dest = out / f"{cls}_{n:03d}_{dd.clean_name(title)}.jpg"
                tmp.replace(dest)
                state["count"] += 1
                print(f"    + {dest.name}")
                time.sleep(0.12)

        for cat in cfg["cats"]:
            if state["count"] >= cfg["target"]:
                break
            try:
                pages = dd.category_files(cat)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {cat}: {exc}")
                continue
            print(f"  - {cat}: {len(pages)}")
            consume(pages)
        for q in cfg["queries"]:
            if state["count"] >= cfg["target"]:
                break
            try:
                pages = dd.search_files(q)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {q}: {exc}")
                continue
            print(f"  - '{q}': {len(pages)}")
            consume(pages)
        print(f"[{cls}] total {state['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
