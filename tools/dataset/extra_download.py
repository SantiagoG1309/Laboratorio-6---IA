"""Descarga imagenes adicionales para completar el dataset y evita duplicados."""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_dataset as dd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"

EXTRA = {
    "trucha": {
        "cats": [
            "Category:Rainbow trout",
            "Category:Oncorhynchus mykiss in aquariums",
            "Category:Trout",
            "Category:Oncorhynchus clarkii",
            "Category:Salmo trutta",
        ],
        "queries": ["rainbow trout", "trout fish underwater", "trout aquaculture"],
        "target": 32,
    },
    "mojarra": {
        "cats": [
            "Category:Oreochromis mossambicus",
            "Category:Oreochromis",
            "Category:Sarotherodon",
            "Category:Tilapia",
            "Category:Red tilapia",
        ],
        "queries": ["tilapia fish", "Oreochromis mossambicus", "mojarra fish", "tilapia aquaculture"],
        "target": 32,
    },
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    for cls, cfg in EXTRA.items():
        out = DS / cls
        out.mkdir(parents=True, exist_ok=True)
        files = [p for p in out.iterdir() if p.suffix.lower() in dd.VALID_EXT]
        hashes = {md5(p) for p in files}
        state = {"count": len(files)}
        print(f"[{cls}] inicio con {state['count']} / {cfg['target']}")

        def consume(pages: list[dict]) -> None:
            for page in pages:
                if state["count"] >= cfg["target"]:
                    return
                title = page.get("title", "")
                if not dd.looks_like_fish_photo(title):
                    continue
                info = (page.get("imageinfo") or [{}])[0]
                mime = info.get("mime", "")
                if mime not in ("image/jpeg", "image/png"):
                    continue
                if info.get("width", 0) < 500 or info.get("height", 0) < 300:
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
                n = state["count"] + 1
                dest = out / f"{cls}_{n:03d}_{dd.clean_name(title)}{tmp.suffix or '.jpg'}"
                tmp.replace(dest)
                state["count"] += 1
                print(f"    + {dest.name}")
                time.sleep(0.15)

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
