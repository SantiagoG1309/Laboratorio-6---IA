# build.ps1 — Compila AquaCNN en un ejecutable nativo de Windows.
#
#   .\build.ps1            # compila en dist\AquaCNN
#   .\build.ps1 -Clean     # limpia build\ y dist\ antes de compilar
#   .\build.ps1 -Zip       # además genera dist\AquaCNN_Windows.zip
#
param(
    [switch]$Clean,
    [switch]$Zip
)

# "Continue" + comprobación explícita de $LASTEXITCODE: PyInstaller escribe su
# progreso por stderr y con "Stop" PowerShell abortaba el script aunque la
# compilación fuese correcta.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
    py -3 -m venv .venv
}

Write-Host "Instalando dependencias..." -ForegroundColor Cyan
# 2>&1 | Out-Host evita que los mensajes de pip/PyInstaller en stderr se
# interpreten como errores nativos y aborten el script con ErrorActionPreference=Stop.
& $python -m pip install -q -r requirements.txt 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "pip falló con código $LASTEXITCODE" }
& $python -m pip install -q "pyinstaller==6.22.3" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "pip falló con código $LASTEXITCODE" }

$icon = Join-Path $PSScriptRoot "ui\assets\aquacnn.ico"
if (-not (Test-Path -LiteralPath $icon)) {
    & $python tools\make_icon.py
}

if ($Clean) {
    Remove-Item -LiteralPath build, dist -Recurse -Force -ErrorAction SilentlyContinue
}

$model = Join-Path $PSScriptRoot "Modelos entrenados\AquaCNN_cnn_peces.npz"
if (-not (Test-Path -LiteralPath $model)) {
    throw "Falta el modelo preentrenado: $model. La aplicación debe distribuirse con él."
}

Write-Host "Compilando con PyInstaller..." -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean AquaCNN.spec 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller falló con código $LASTEXITCODE"
}

$exe = Join-Path $PSScriptRoot "dist\AquaCNN\AquaCNN.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw "No se generó el ejecutable esperado: $exe"
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "LEEME.md") `
    -Destination (Join-Path $PSScriptRoot "dist\AquaCNN\LEEME.md") -Force

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "README.md") `
    -Destination (Join-Path $PSScriptRoot "dist\AquaCNN\README.md") -Force

# Capturas usadas por el README (para que se vean también sin conexión).
$captures = Join-Path $PSScriptRoot "docs\capturas"
if (Test-Path -LiteralPath $captures) {
    $docsDest = Join-Path $PSScriptRoot "dist\AquaCNN\docs"
    New-Item -ItemType Directory -Path $docsDest -Force | Out-Null
    Copy-Item -LiteralPath $captures -Destination $docsDest -Recurse -Force
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "COMO_USAR.txt") `
    -Destination (Join-Path $PSScriptRoot "dist\AquaCNN\COMO_USAR.txt") -Force

$size = [math]::Round((Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "dist\AquaCNN") -Recurse -File |
    Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host ""
Write-Host "Listo: $exe" -ForegroundColor Green
Write-Host "Tamaño de la carpeta: $size MB" -ForegroundColor Green
Write-Host "Distribuye la carpeta dist\AquaCNN completa (no solo el .exe)." -ForegroundColor Yellow

if ($Zip) {
    $zipPath = Join-Path $PSScriptRoot "dist\AquaCNN_Windows.zip"
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
    # El ZIP contiene los archivos en la raíz (sin carpeta AquaCNN envolvente)
    # para que al descomprimir el .exe quede directamente accesible.
    Compress-Archive -Path (Join-Path $PSScriptRoot "dist\AquaCNN\*") -DestinationPath $zipPath
    if (-not (Test-Path -LiteralPath $zipPath)) { throw "No se generó el ZIP: $zipPath" }
    Write-Host "ZIP: $zipPath" -ForegroundColor Green
}

exit 0
