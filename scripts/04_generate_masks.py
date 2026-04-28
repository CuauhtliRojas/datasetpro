# scripts/04_generate_masks.py
"""
Genera las máscaras binarias por sustracción automática.

Por cada par (real, fake) produce DOS máscaras:
  - fake_mask:     región manipulada = blanco (255), resto = negro (0)
  - original_mask: región auténtica  = blanco (255), resto = negro (0)
                   (es el inverso de fake_mask, restringido al rostro)

Justificación: §8.5.4 del documento — segmentación binaria,
               §8.8.3 — acotamiento próximo con tolerancia morfológica.

Requiere: CPU es suficiente — pero se ejecuta DESPUÉS del script 02
Entrada:  data/raw/real/      (imágenes originales)
          data/raw/fake_swap/ (imágenes manipuladas)
Salida:   data/Train_D/fake_mask/
          data/Train_D/original_mask/
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Configuración ─────────────────────────────────────────────────────────────
REAL_DIR     = Path("data/raw/real")
FAKE_DIR     = Path("data/raw/fake_swap")
FAKE_MASK_DIR    = Path("data/Train_D/fake_mask")
ORIG_MASK_DIR    = Path("data/Train_D/original_mask")

FAKE_MASK_DIR.mkdir(parents=True, exist_ok=True)
ORIG_MASK_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de la máscara (§8.8.3 — tolerancia morfológica)
THRESHOLD    = 15     # Diferencia mínima de píxel para considerarlo manipulado
DILATION_PX  = 18     # Expansión morfológica para cubrir bordes de transición
OUTPUT_SIZE  = (128, 128)  # Igual que train_.py usa Resize(128,128)

# ── Detector de rostro ────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detectar_rostro(img_gray, output_size):
    """Devuelve máscara binaria de la región facial detectada."""
    roi = np.zeros(img_gray.shape[:2], dtype=np.uint8)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        # Sin detección: usar zona central como fallback (§8.11.3)
        h, w = img_gray.shape
        margin_y, margin_x = h // 5, w // 5
        roi[margin_y:h - margin_y, margin_x:w - margin_x] = 255
    else:
        for (x, y, w, h) in faces:
            roi[y:y+h, x:x+w] = 255
    return roi

def generar_mascaras(real_path, fake_path):
    """
    Devuelve (fake_mask, original_mask) como arrays numpy uint8.
    Ambas en OUTPUT_SIZE, valores 0 o 255.
    """
    real = cv2.imread(str(real_path))
    fake = cv2.imread(str(fake_path))

    if real is None or fake is None:
        return None, None

    # Igualar tamaños antes de restar
    h, w = real.shape[:2]
    fake_resized = cv2.resize(fake, (w, h))

    # ── Sustracción ───────────────────────────────────────────────────────────
    diff = cv2.absdiff(real, fake_resized)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(diff_gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    # ── Dilatación morfológica (acotamiento próximo, §8.8.3) ──────────────────
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATION_PX, DILATION_PX)
    )
    manipulated_region = cv2.dilate(binary, kernel, iterations=2)

    # ── Restringir al rostro (§8.1.3 — solo región facial) ───────────────────
    real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    face_roi = detectar_rostro(real_gray, (w, h))
    fake_mask = cv2.bitwise_and(manipulated_region, face_roi)

    # ── Original mask = zona auténtica del rostro (inverso dentro del ROI) ────
    # Píxeles del rostro que NO fueron manipulados
    original_mask = cv2.bitwise_and(
        cv2.bitwise_not(fake_mask),
        face_roi
    )

    # ── Redimensionar a OUTPUT_SIZE ───────────────────────────────────────────
    fake_mask_out = cv2.resize(fake_mask,     OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    orig_mask_out = cv2.resize(original_mask, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)

    return fake_mask_out, orig_mask_out


# ── Procesar todos los pares ──────────────────────────────────────────────────
fake_images = sorted(FAKE_DIR.glob("*.png"))
print(f"[1/2] Pares encontrados: {len(fake_images)}")

if len(fake_images) == 0:
    print("ERROR: No hay imágenes en data/raw/fake_swap/")
    print("Ejecuta primero: python scripts/02_generate_swaps.py  (en Laptop)")
    exit(1)

vacias    = 0
exitosas  = 0
errores   = 0

print("[2/2] Generando máscaras...")
for fake_path in tqdm(fake_images, desc="Máscaras"):
    real_path = REAL_DIR / fake_path.name

    if not real_path.exists():
        errores += 1
        continue

    fake_mask, orig_mask = generar_mascaras(real_path, fake_path)

    if fake_mask is None:
        errores += 1
        continue

    cv2.imwrite(str(FAKE_MASK_DIR / fake_path.name), fake_mask)
    cv2.imwrite(str(ORIG_MASK_DIR / fake_path.name), orig_mask)

    if fake_mask.max() == 0:
        vacias += 1
    else:
        exitosas += 1

# ── Resumen ───────────────────────────────────────────────────────────────────
print(f"\nResumen:")
print(f"  Máscaras con región detectada: {exitosas}")
print(f"  Máscaras vacías (swap mínimo): {vacias}")
print(f"  Errores:                       {errores}")
print(f"  Archivos en fake_mask/:    {len(list(FAKE_MASK_DIR.glob('*.png')))}")
print(f"  Archivos en original_mask/:{len(list(ORIG_MASK_DIR.glob('*.png')))}")