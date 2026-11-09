"""Descarga fotos de peces desde iNaturalist (observaciones de grado investigación).

API publica, sin clave. Solo fotos con licencia Creative Commons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"
API = "https://api.inaturalist.org/v1/observations"
UA = "Lab6FishClassifier/1.0 (educational)"

TAXA = {
    "trucha": ["Oncorhynchus mykiss"],
    "mojarra": ["Oreochromis niloticus", "Oreochromis mossambicus"],
    "pavon": ["Cichla ocellaris", "Cichla temensis"],
}
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"


def get_json(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            blob = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {exc}")
        return False
    if len(blob) < 12000:
        return False
    if not (blob[:2] == b"\xff\xd8" or blob[:4] == b"\x89PNG"):
        return False
    dest.write_bytes(blob)
    return True


def clean(text: str, limit: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:limit].strip("_")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=35)
    args = ap.parse_args()

    credits = ["Fotografias adicionales: iNaturalist (www.inaturalist.org)", ""]
    for cls, taxa in TAXA.items():
        out = DS / cls
        out.mkdir(parents=True, exist_ok=True)
        files = [p for p in out.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        hashes = {hashlib.md5(p.read_bytes()).hexdigest() for p in files}
        count = len(files)
        print(f"[{cls}] inicio {count} / {args.target}")
        for taxon in taxa:
            if count >= args.target:
                break
            page = 1
            while count < args.target and page <= 5:
                params = {
                    "taxon_name": taxon,
                    "photos": "true",
                    "quality_grade": "research",
                    "photo_license": LICENSES,
                    "per_page": "50",
                    "page": str(page),
                    "order_by": "votes",
                    "order": "desc",
                }
                try:
                    data = get_json(params)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {taxon}: {exc}")
                    break
                results = data.get("results", [])
                if not results:
                    break
                print(f"  - {taxon} p{page}: {len(results)} obs")
                for obs in results:
                    if count >= args.target:
                        break
                    for photo in obs.get("photos", []):
                        if count >= args.target:
                            break
                        if photo.get("license_code") not in ("cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"):
                            continue
                        url = photo.get("url", "").replace("/square.", "/medium.")
                        if not url:
                            continue
                        n = count + 1
                        dest = out / f"{cls}_{n:03d}_inat_{clean(taxon)}_{obs.get('id')}.jpg"
                        print(f"    + {dest.name}")
                        if download(url, dest):
                            h = hashlib.md5(dest.read_bytes()).hexdigest()
                            if h in hashes:
                                dest.unlink(missing_ok=True)
                                continue
                            hashes.add(h)
                            credits.append(
                                f"{cls}/{dest.name}\t{obs.get('user', {}).get('login', '?')}\t"
                                f"{photo.get('license_code', '?')}\thttps://www.inaturalist.org/observations/{obs.get('id')}"
                            )
                            count += 1
                        time.sleep(0.1)
                page += 1
        print(f"[{cls}] total {count}")

    (DS / "CREDITOS_INATURALIST.txt").write_text("\n".join(credits), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
