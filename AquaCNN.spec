# -*- mode: python ; coding: utf-8 -*-
"""Empaquetado de AquaCNN con PyInstaller (modo carpeta, arranque rápido).

Generar con:  pyinstaller --noconfirm --clean AquaCNN.spec
Resultado:    dist/AquaCNN/AquaCNN.exe
"""
from pathlib import Path

ROOT = Path(SPECPATH)  # noqa: F821 - definido por PyInstaller

datas = [
    (str(ROOT / "Dataset Imagenes"), "Dataset Imagenes"),
    (str(ROOT / "ui" / "assets"), "ui/assets"),
    (str(ROOT / "README.md"), "."),
    (str(ROOT / "requirements.txt"), "."),
]
# Modelo preentrenado: se incluye dentro del ejecutable para que la app tenga
# resultados desde el primer arranque (se copia junto al .exe la primera vez).
models = ROOT / "Modelos entrenados"
if models.exists() and any(models.glob("*.npz")):
    datas.append((str(models), "Modelos entrenados"))
credits = ROOT / "Dataset Imagenes" / "CREDITOS.txt"
if credits.exists():
    datas.append((str(credits), "Dataset Imagenes"))

hiddenimports = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "openpyxl",
    "openpyxl.styles",
    "openpyxl.utils",
    "openpyxl.cell._writer",
]

excludes = [
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "setuptools",
    "pip",
    "scipy",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtPdf",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.Qt3DCore",
    "PySide6.QtMultimedia",
    "PySide6.QtCharts",
]

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

# --- Recorte de binarios no utilizados (reduce ~90 MB) -----------------------
_DROP_PARTS = (
    "opencv_videoio_ffmpeg",
    "qt6pdf",
    "qt6quick",
    "qt6qml",
    "qt6virtualkeyboard",
    "qtvirtualkeyboard",
    "/scipy/",
    "scipy.libs",
)


def _keep(entry) -> bool:
    name = entry[0].replace("\\", "/").lower()
    return not any(part in name for part in _DROP_PARTS)


a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if "scipy" not in d[0].replace("\\", "/").lower()]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AquaCNN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "ui" / "assets" / "aquacnn.ico"),
    version=str(ROOT / "tools" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AquaCNN",
)
