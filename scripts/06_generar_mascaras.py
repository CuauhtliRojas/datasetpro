# scripts/04_generar_mascaras.py
# =============================================================================
# PASO 4 — Generación automática de máscaras binarias
# =============================================================================
#
# QUÉ HACE:
#   Por cada par (imagen_real, imagen_fake) genera dos máscaras binarias:
#   - fake_mask:     región manipulada por el swap = blanco (255)
#   - original_mask: región auténtica = blanco (255)
#
# MÉTODO (8.5.4, 8.8.3):
#   1. Sustracción absoluta: absdiff(real, fake) píxel a píxel
#   2. Umbralización: diferencia > THRESHOLD → región manipulada
#   3. Dilatación morfológica: expansión para cubrir bordes de transición
#      (produce el "acotamiento próximo" descrito en 8.8.3 y 5)
#   4. original_mask = inverso lógico de fake_mask
#
# NOTA DE RESOLUCIÓN:
#   MODO PRUEBA (128px):     THRESHOLD=5,  sin restricción facial por Haar
#   MODO PRODUCCIÓN (256px+): THRESHOLD=15, con restricción facial por Haar
#   El detector Haar de OpenCV falla con imágenes de 128px — se omite en prueba.
#
# ENTRADA:  data/raw/real/       — imágenes reales originales
#           data/raw/fake_swap/  — imágenes con rostro intercambiado
# SALIDA:   data/Train_D/fake_mask/     — máscara región manipulada
#           data/Train_D/original_mask/ — máscara región auténtica
# =============================================================================

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Rutas ─────────────────────────────────────────────────────────────────────
DIRECTORIO_REAL      = Path("data/raw/real")
DIRECTORIO_FAKE      = Path("data/raw/fake_swap")
DIRECTORIO_FAKE_MASK = Path("data/Train_D/fake_mask")
DIRECTORIO_ORIG_MASK = Path("data/Train_D/original_mask")

DIRECTORIO_FAKE_MASK.mkdir(parents=True, exist_ok=True)
DIRECTORIO_ORIG_MASK.mkdir(parents=True, exist_ok=True)

# ── Parámetros — MODO PRUEBA (128px) ─────────────────────────────────────────
#UMBRAL      = 5    # Bajo porque el blending de inswapper es casi perfecto a 128px
#DILATACION  = 8    # Proporcional a la resolución (equivale a 18px en 256px)
#TAMANIO     = (128, 128)

# ── Parámetros — MODO PRODUCCIÓN (descomentar al escalar a 256px+) ───────────
UMBRAL      = 15
DILATACION  = 18
TAMANIO     = (256, 256)
detector_haar = cv2.CascadeClassifier(
     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def generar_par_mascaras(ruta_real, ruta_fake):
    """
    Genera fake_mask y original_mask para un par de imágenes.

    Parámetros:
        ruta_real: Path — imagen auténtica original
        ruta_fake: Path — imagen con rostro intercambiado

    Retorna:
        (fake_mask, original_mask) como arrays numpy uint8 con valores 0 o 255
        (None, None) si hay error de lectura
    """
    real = cv2.imread(str(ruta_real))
    fake = cv2.imread(str(ruta_fake))

    if real is None or fake is None:
        return None, None

    # Igualar tamaños antes de restar
    h, w    = real.shape[:2]
    fake_r  = cv2.resize(fake, (w, h))

    # 1. Sustracción absoluta píxel a píxel
    diferencia      = cv2.absdiff(real, fake_r)
    diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

    # 2. Umbralización — solo diferencias mayores al umbral
    _, binario = cv2.threshold(diferencia_gris, UMBRAL, 255, cv2.THRESH_BINARY)

    # 3. Dilatación morfológica — acotamiento próximo (8.8.3)
    #    Expande la región detectada para cubrir los bordes de transición
    #    entre contenido real y sintético
    nucleo    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATACION, DILATACION))
    fake_mask = cv2.dilate(binario, nucleo, iterations=2)

    # 4. original_mask = zona NO manipulada (inverso lógico)
    original_mask = cv2.bitwise_not(fake_mask)

    # 5. Redimensionar a tamaño de salida
    fake_out = cv2.resize(fake_mask,     TAMANIO, interpolation=cv2.INTER_NEAREST)
    orig_out = cv2.resize(original_mask, TAMANIO, interpolation=cv2.INTER_NEAREST)

    return fake_out, orig_out


# ── Procesar todos los pares ──────────────────────────────────────────────────
imagenes_fake = sorted(DIRECTORIO_FAKE.glob("*.png"))
print(f"[1/2] Pares encontrados: {len(imagenes_fake)}")

if len(imagenes_fake) == 0:
    print("ERROR: No hay imágenes en data/raw/fake_swap/")
    print("Solución: ejecutar primero  uv run python scripts/02_generar_swaps.py")
    exit(1)

exitosas = 0
vacias   = 0
errores  = 0

print("[2/2] Generando máscaras binarias...")
for ruta_fake in tqdm(imagenes_fake, desc="Máscaras", unit="img"):
    ruta_real = DIRECTORIO_REAL / ruta_fake.name

    if not ruta_real.exists():
        errores += 1
        continue

    fake_mask, orig_mask = generar_par_mascaras(ruta_real, ruta_fake)

    if fake_mask is None:
        errores += 1
        continue

    cv2.imwrite(str(DIRECTORIO_FAKE_MASK / ruta_fake.name), fake_mask)
    cv2.imwrite(str(DIRECTORIO_ORIG_MASK / ruta_fake.name), orig_mask)

    if fake_mask.max() == 0:
        vacias += 1
    else:
        exitosas += 1

# ── Resumen ───────────────────────────────────────────────────────────────────
print(f"\nResumen:")
print(f"  Máscaras con región detectada: {exitosas}")
print(f"  Máscaras vacías:               {vacias}")
print(f"  Errores:                       {errores}")
print(f"  Archivos en fake_mask/:        {len(list(DIRECTORIO_FAKE_MASK.glob('*.png')))}")
print(f"  Archivos en original_mask/:    {len(list(DIRECTORIO_ORIG_MASK.glob('*.png')))}")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/05_ensamblar_dataset.py")