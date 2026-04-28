# scripts/02_generate_swaps.py
"""
Genera 1,000 imágenes fake usando InsightFace (inswapper_128).
Por cada imagen real, toma un rostro fuente aleatorio diferente
y produce la imagen con el rostro intercambiado.

Requiere: GPU NVIDIA — ejecutar en Laptop RTX 4050
Entrada:  data/raw/real/         (1,000 PNG)
Salida:   data/raw/fake_swap/    (1,000 PNG con mismo nombre)
"""

import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

# ── Configuración ─────────────────────────────────────────────────────────────
REAL_DIR  = Path("data/raw/real")
FAKE_DIR  = Path("data/raw/fake_swap")
FAKE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

# ── Inicializar modelo ────────────────────────────────────────────────────────
print("[1/4] Inicializando InsightFace (se descarga inswapper_128.onnx si es primera vez)...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(
    "inswapper_128.onnx",
    download=True,
    download_zip=True
)

# ── Cargar lista de imágenes ──────────────────────────────────────────────────
images = sorted(REAL_DIR.glob("*.png"))
print(f"[2/4] Imágenes reales encontradas: {len(images)}")

if len(images) == 0:
    print("ERROR: No hay imágenes en data/raw/real/")
    print("Ejecuta primero: python scripts/01_download_ffhq.py")
    exit(1)

# ── Generar fakes ─────────────────────────────────────────────────────────────
print("[3/4] Generando face swaps...")
exitosos   = 0
sin_rostro = 0
errores    = 0

for target_path in tqdm(images, desc="SwapFace"):
    dest_path = FAKE_DIR / target_path.name

    # Si ya existe, saltar (permite reanudar si se interrumpe)
    if dest_path.exists():
        exitosos += 1
        continue

    # Elegir fuente aleatoria diferente al target
    candidates = [p for p in images if p != target_path]
    source_path = random.choice(candidates)

    try:
        target_img = cv2.imread(str(target_path))
        source_img = cv2.imread(str(source_path))

        target_faces = app.get(target_img)
        source_faces = app.get(source_img)

        # Si alguna imagen no tiene rostro detectable, guardar original sin swap
        if not target_faces or not source_faces:
            cv2.imwrite(str(dest_path), target_img)
            sin_rostro += 1
            continue

        # Aplicar swap del primer rostro detectado
        result = target_img.copy()
        result = swapper.get(result, target_faces[0], source_faces[0], paste_back=True)
        cv2.imwrite(str(dest_path), result)
        exitosos += 1

    except Exception as e:
        # En caso de error, guardar imagen original para no romper el pipeline
        cv2.imwrite(str(dest_path), cv2.imread(str(target_path)))
        errores += 1

# ── Resumen ───────────────────────────────────────────────────────────────────
print(f"\n[4/4] Resumen:")
print(f"  Swaps exitosos:        {exitosos}")
print(f"  Sin rostro detectable: {sin_rostro}")
print(f"  Errores:               {errores}")
print(f"  Total en fake_swap/:   {len(list(FAKE_DIR.glob('*.png')))}")