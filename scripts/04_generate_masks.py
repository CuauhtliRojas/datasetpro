# scripts/04_generate_masks.py
"""
Genera máscaras binarias por sustracción imagen_real − imagen_fake.
Aplica dilatación morfológica para obtener el 'acotamiento próximo'
descrito explícitamente en §5 y §8.8.3.
Valor 1 = región alterada, valor 0 = auténtica (§8.5.4).
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_mask(real_path: Path, fake_path: Path, out_path: Path,
                  threshold: int = 15, dilation_px: int = 12):
    real = cv2.imread(str(real_path))
    fake = cv2.imread(str(fake_path))
    if real is None or fake is None:
        return False

    # Redimensionar a 256x256 — divisible por 16 para U-Net (§8.6.2)
    real = cv2.resize(real, (256, 256))
    fake = cv2.resize(fake, (256, 256))

    # Sustracción en escala de grises
    diff = cv2.absdiff(real, fake)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Umbralización
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Dilatación morfológica — genera el 'acotamiento próximo' (§8.8.3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
    mask = cv2.dilate(binary, kernel, iterations=2)

    # Restricción a región facial usando detector haar (§8.1.3 — solo rostro)
    gray_real = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray_real, 1.1, 4)
    face_roi = np.zeros((256, 256), dtype=np.uint8)
    for (x, y, w, h) in faces:
        face_roi[y:y+h, x:x+w] = 255

    # Máscara final restringida al rostro
    mask_final = cv2.bitwise_and(mask, face_roi)

    # Guardar como binario normalizado: 0 o 1 (§8.5.4)
    cv2.imwrite(str(out_path), (mask_final > 0).astype(np.uint8) * 255)
    return True

SWAP_FAKE   = Path("data/raw/fake_swap")
INPAINT_FAKE = Path("data/raw/fake_inpaint")
REAL_DIR    = Path("data/raw/real")
MASK_DIR    = Path("data/processed/masks")
MASK_DIR.mkdir(parents=True, exist_ok=True)

for fake_dir in [SWAP_FAKE, INPAINT_FAKE]:
    for fake_path in tqdm(fake_dir.glob("*.png")):
        real_path = REAL_DIR / fake_path.name
        if real_path.exists():
            generate_mask(real_path, fake_path, MASK_DIR / fake_path.name)