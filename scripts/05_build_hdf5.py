# scripts/05_assemble_dataset.py
"""
Mueve las imágenes fake a Train_D/images/ y verifica
que los tres directorios tienen exactamente el mismo
número de archivos con los mismos nombres.

Train_D/images/        ← imagen fake (entrada al modelo)
Train_D/fake_mask/     ← ya generada por script 04
Train_D/original_mask/ ← ya generada por script 04

Requiere: CPU — ejecutar después del script 04
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# ── Rutas ─────────────────────────────────────────────────────────────────────
FAKE_SRC   = Path("data/raw/fake_swap")
IMAGES_DIR = Path("data/Train_D/images")
FAKE_MASK  = Path("data/Train_D/fake_mask")
ORIG_MASK  = Path("data/Train_D/original_mask")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Copiar imágenes fake a Train_D/images/ ────────────────────────────────────
fake_files = sorted(FAKE_SRC.glob("*.png"))
print(f"[1/3] Copiando {len(fake_files)} imágenes fake a Train_D/images/ ...")

for src in tqdm(fake_files, desc="Copiando"):
    dst = IMAGES_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)

# ── Verificación de consistencia ──────────────────────────────────────────────
print("\n[2/3] Verificando consistencia del dataset...")

images_set = {f.name for f in IMAGES_DIR.glob("*.png")}
fmask_set  = {f.name for f in FAKE_MASK.glob("*.png")}
omask_set  = {f.name for f in ORIG_MASK.glob("*.png")}

print(f"  images/:        {len(images_set):>6} archivos")
print(f"  fake_mask/:     {len(fmask_set):>6} archivos")
print(f"  original_mask/: {len(omask_set):>6} archivos")

# Archivos que faltan en alguna carpeta
solo_en_images   = images_set - fmask_set - omask_set
faltan_en_images = (fmask_set | omask_set) - images_set

if solo_en_images:
    print(f"\n  ADVERTENCIA: {len(solo_en_images)} imágenes sin máscara correspondiente")
if faltan_en_images:
    print(f"  ADVERTENCIA: {len(faltan_en_images)} máscaras sin imagen correspondiente")

# Conjunto válido (tiene las tres carpetas)
validos = images_set & fmask_set & omask_set

# ── Resumen final ─────────────────────────────────────────────────────────────
print(f"\n[3/3] Resultado:")
print(f"  Tripletas completas (imagen + 2 máscaras): {len(validos)}")

if len(validos) == len(images_set) == len(fmask_set) == len(omask_set):
    print("  ✓ Dataset consistente. Listo para entrenamiento.")
    print(f"\n  Siguiente paso (en Laptop):")
    print(f"  python src/train_.py")
else:
    print("  ✗ Dataset inconsistente. Revisa los pasos anteriores.")