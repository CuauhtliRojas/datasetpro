# scripts/01_descargar_ffhq.py
# =============================================================================
# PASO 1 — Descarga de imágenes reales FFHQ
# =============================================================================
#
# QUÉ HACE:
#   Descarga el dataset FFHQ (Flickr-Faces-HQ) desde Kaggle y selecciona
#   aleatoriamente 500 imágenes para usarlas como base auténtica..
#
# POR QUÉ FFHQ (8.8.1 del documento):
#   - Diversidad demográfica real sin sesgo de identidad de actor
#   - Alta resolución original (1024px) con señales de alta frecuencia (8.5.2)
#   - Licencia de investigación compatible
#
# REQUISITOS:
#   - Cuenta de Kaggle con API key configurada en ~/.kaggle/kaggle.json
#   - Conexión a internet (~2 GB de descarga)
#
# ENTRADA:  Ninguna
# SALIDA:   data/raw/real/  — 500 PNG numerados (00000.png ... 00499.png)
#
# CÓMO REANUDAR SI SE INTERRUMPE:
#   Borra data/raw/real/ y vuelve a ejecutar el script.
# =============================================================================

import subprocess
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_REAL = Path("data/raw/real")
DIRECTORIO_TEMP = Path("data/raw/_kaggle_temp")
DIRECTORIO_REAL.mkdir(parents=True, exist_ok=True)
DIRECTORIO_TEMP.mkdir(parents=True, exist_ok=True)

TOTAL_IMAGENES = 500
SEMILLA = 42  # Semilla fija para reproducibilidad (8.8.3)
random.seed(SEMILLA)

# ── Descarga desde Kaggle ─────────────────────────────────────────────────────
print("[1/3] Descargando FFHQ desde Kaggle...")
print("      (Primera ejecución: ~2 GB de descarga)")
resultado = subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "greatgamedota/ffhq-face-data-set",
    "-p", str(DIRECTORIO_TEMP),
    "--unzip"
], capture_output=False)

if resultado.returncode != 0:
    print("ERROR: Falló la descarga de Kaggle.")
    print("Solución: Verifica que existe C:\\Users\\TU_USUARIO_DE_PC\\.kaggle\\kaggle.json")
    exit(1)

# ── Buscar imágenes descargadas ───────────────────────────────────────────────
print("[2/3] Buscando imágenes descargadas...")
todas = (list(DIRECTORIO_TEMP.rglob("*.png")) +
         list(DIRECTORIO_TEMP.rglob("*.jpg")) +
         list(DIRECTORIO_TEMP.rglob("*.jpeg")))

print(f"      Encontradas: {len(todas)} imágenes en el dataset completo")

if len(todas) == 0:
    print("ERROR: No se encontraron imágenes en la descarga.")
    print("Contenido de la carpeta temporal:")
    for f in DIRECTORIO_TEMP.rglob("*"):
        print(f"  {f}")
    exit(1)

# ── Seleccionar y guardar ─────────────────────────────────────────────────────
n = min(TOTAL_IMAGENES, len(todas))
seleccionadas = random.sample(todas, n)

print(f"[3/3] Guardando {n} imágenes en data/raw/real/ ...")
errores = 0
for i, ruta_origen in enumerate(tqdm(seleccionadas, desc="Copiando", unit="img")):
    destino = DIRECTORIO_REAL / f"{i:05d}.png"
    try:
        img = Image.open(ruta_origen).convert("RGB")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(destino, "PNG")
    except Exception as e:
        print(f"  Saltando {ruta_origen.name}: {e}")
        errores += 1

# ── Limpieza y resumen ────────────────────────────────────────────────────────
shutil.rmtree(DIRECTORIO_TEMP)

total = len(list(DIRECTORIO_REAL.glob("*.png")))
print(f"\nResultado: {total} imágenes guardadas en data/raw/real/")
if errores:
    print(f"Advertencia: {errores} imágenes no se pudieron procesar.")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/02_generar_swaps.py")