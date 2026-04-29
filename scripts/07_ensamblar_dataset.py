# scripts/05_ensamblar_dataset.py
# =============================================================================
# PASO 5  — Ensamblado final del dataset Train_D/
# =============================================================================
#
# QUÉ HACE:
#   Copia las imágenes fake a Train_D/images/ y verifica que las tres
#   carpetas requeridas por train.py tengan exactamente los mismos archivos.
#
# ESTRUCTURA QUE GENERA:
#   Train_D/
#   ├── images/        ← imagen fake (entrada al modelo DualSegmentationModel)
#   ├── fake_mask/     ← máscara región manipulada (ground truth decoder 1)
#   └── original_mask/ ← máscara región auténtica  (ground truth decoder 2)
#
# ENTRADA:  data/raw/fake_swap/          — imágenes falsas generadas
#           data/Train_D/fake_mask/      — ya generadas por paso 4
#           data/Train_D/original_mask/  — ya generadas por paso 4
# SALIDA:   data/Train_D/images/         — listo para entrenamiento
# =============================================================================

import shutil
from pathlib import Path
from tqdm import tqdm

# ── Rutas ─────────────────────────────────────────────────────────────────────
ORIGEN_FAKE    = Path("data/raw/fake_swap")
DIRECTORIO_IMG = Path("data/Train_D/images")
DIRECTORIO_FM  = Path("data/Train_D/fake_mask")
DIRECTORIO_OM  = Path("data/Train_D/original_mask")

DIRECTORIO_IMG.mkdir(parents=True, exist_ok=True)

# ── Copiar imágenes fake a Train_D/images/ ────────────────────────────────────
archivos_fake = sorted(ORIGEN_FAKE.glob("*.png"))
print(f"[1/3] Copiando {len(archivos_fake)} imágenes fake a Train_D/images/ ...")

for origen in tqdm(archivos_fake, desc="Copiando", unit="img"):
    destino = DIRECTORIO_IMG / origen.name
    if not destino.exists():
        shutil.copy2(origen, destino)

# ── Verificación de consistencia ──────────────────────────────────────────────
print("\n[2/3] Verificando consistencia del dataset...")

conjunto_img = {f.name for f in DIRECTORIO_IMG.glob("*.png")}
conjunto_fm  = {f.name for f in DIRECTORIO_FM.glob("*.png")}
conjunto_om  = {f.name for f in DIRECTORIO_OM.glob("*.png")}

print(f"  images/:        {len(conjunto_img):>6} archivos")
print(f"  fake_mask/:     {len(conjunto_fm):>6} archivos")
print(f"  original_mask/: {len(conjunto_om):>6} archivos")

sin_mascara     = conjunto_img - conjunto_fm - conjunto_om
sin_imagen      = (conjunto_fm | conjunto_om) - conjunto_img
tripletas_validas = conjunto_img & conjunto_fm & conjunto_om

if sin_mascara:
    print(f"\n  ADVERTENCIA: {len(sin_mascara)} imágenes sin máscara correspondiente")
if sin_imagen:
    print(f"  ADVERTENCIA: {len(sin_imagen)} máscaras sin imagen correspondiente")

# ── Resultado final ───────────────────────────────────────────────────────────
print(f"\n[3/3] Resultado:")
print(f"  Tripletas completas (imagen + 2 máscaras): {len(tripletas_validas)}")

if len(tripletas_validas) == len(conjunto_img) == len(conjunto_fm) == len(conjunto_om):
    print("  ✓ Dataset consistente. Listo para entrenamiento.")
    print(f"\nSIGUIENTE PASO:")
    print(f"  uv run python scripts/06_verificar_dataset.py")
else:
    print("  ✗ Dataset inconsistente. Revisa los pasos anteriores.")
    print("    Ejecuta: uv run python scripts/06_verificar_dataset.py")