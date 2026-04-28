# scripts/01_download_ffhq.py
"""
Descarga 1,000 imágenes de FFHQ desde Kaggle.
Guarda en data/raw/real/ como PNG numerados (00000.png ... 00999.png).
"""
import subprocess
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAW_REAL = Path("data/raw/real")
RAW_REAL.mkdir(parents=True, exist_ok=True)

TEMP_DIR = Path("data/raw/_kaggle_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 1000

# ── Descarga ──────────────────────────────────────────────────────────────────
print("[1/3] Descargando FFHQ thumbnails desde Kaggle...")
result = subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "greatgamedota/ffhq-face-data-set",
    "-p", str(TEMP_DIR),
    "--unzip"
], capture_output=False)

if result.returncode != 0:
    print("ERROR: Falló la descarga. Verifica tu kaggle.json")
    exit(1)

# ── Buscar imágenes descargadas ───────────────────────────────────────────────
print("[2/3] Buscando imágenes descargadas...")
all_images = (list(TEMP_DIR.rglob("*.png")) +
              list(TEMP_DIR.rglob("*.jpg")) +
              list(TEMP_DIR.rglob("*.jpeg")))

print(f"      Encontradas: {len(all_images)} imágenes")

if len(all_images) == 0:
    print("ERROR: No se encontraron imágenes.")
    print("Contenido de la carpeta temporal:")
    for f in TEMP_DIR.rglob("*"):
        print(f"  {f}")
    exit(1)

# ── Seleccionar y copiar ──────────────────────────────────────────────────────
random.seed(42)
n = min(TARGET_COUNT, len(all_images))
selected = random.sample(all_images, n)

print(f"[3/3] Copiando {n} imágenes a data/raw/real/ ...")
errores = 0
for i, img_path in enumerate(tqdm(selected)):
    dest = RAW_REAL / f"{i:05d}.png"
    try:
        img = Image.open(img_path).convert("RGB")
        img.save(dest, "PNG")
    except Exception as e:
        print(f"  Saltando {img_path.name}: {e}")
        errores += 1

# ── Limpieza ──────────────────────────────────────────────────────────────────
shutil.rmtree(TEMP_DIR)

total = len(list(RAW_REAL.glob("*.png")))
print(f"\nListo. {total} imágenes guardadas en data/raw/real/")
if errores:
    print(f"Advertencia: {errores} imágenes no se pudieron procesar.")