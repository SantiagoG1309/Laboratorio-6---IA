"""Crea hojas de contacto del dataset para revisión visual."""
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Dataset Imagenes"
OUT = ROOT / "Tema" / "contact"
OUT.mkdir(parents=True, exist_ok=True)

for cls in ("trucha", "mojarra", "pavon"):
    files = sorted([p for p in (DS / cls).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    cols, rows = 5, 4
    per_page = cols * rows
    for start in range(0, len(files), per_page):
        chunk = files[start:start + per_page]
        doc = pymupdf.open()
        page = doc.new_page(width=1000, height=760)
        cw, ch = 190, 150
        for i, f in enumerate(chunk):
            r, c = divmod(i, cols)
            x, y = 10 + c * 198, 30 + r * 182
            try:
                page.insert_image(pymupdf.Rect(x, y, x + cw, y + ch), filename=str(f), keep_proportion=True)
            except Exception as exc:  # noqa: BLE001
                page.insert_text((x + 5, y + 20), f"ERR {exc}", fontsize=6)
            page.insert_text((x, y + ch + 12), f.name[:34], fontsize=6)
        pix = page.get_pixmap(dpi=110)
        pix.save(OUT / f"{cls}_{start // per_page + 1}.png")
        doc.close()
        print("ok", cls, start // per_page + 1)
