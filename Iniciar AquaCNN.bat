@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AquaCNN - Laboratorio 6

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"

if exist "%PY%" goto deps

echo [1/3] Creando entorno virtual de Python...
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv "%VENV%"
) else (
    python -m venv "%VENV%"
)
if not exist "%PY%" goto fail

:deps
echo [2/3] Verificando dependencias...
"%PY%" -c "import PySide6, cv2, numpy, matplotlib" >nul 2>nul
if not errorlevel 1 goto run

echo Instalando dependencias. La primera vez puede tardar varios minutos...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo Reintentando con un espejo alternativo...
    "%PY%" -m pip install -r "%~dp0requirements.txt" --index-url https://mirrors.aliyun.com/pypi/simple
)
if errorlevel 1 goto fail

:run
echo [3/3] Iniciando AquaCNN...
start "" "%VENV%\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0

:fail
echo.
echo ERROR: no se pudo preparar el entorno.
echo Comprueba que Python 3.10 o superior este instalado y disponible en el PATH.
pause
exit /b 1
