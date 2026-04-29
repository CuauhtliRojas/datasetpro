# scripts/06_verificar_dataset.py
# =============================================================================
# DIAGNÓSTICO — Estado actual del dataset en cualquier momento
# =============================================================================
#
# QUÉ HACE:
#   Muestra el estado de todas las carpetas del pipeline y dice cuál es
#   el siguiente paso a ejecutar. Puedes correr este script en cualquier
#   momento y en cualquier dispositivo (no requiere GPU).
# =============================================================================

from pathlib import Path

def contar(carpeta, extension="*.png"):
    p = Path(carpeta)
    if not p.exists():
        return 0, "❌ carpeta no existe"
    n = len(list(p.glob(extension)))
    estado = "✓" if n > 0 else "⚠  vacía"
    return n, estado

print("=" * 60)
print("ESTADO DEL DATASET — Pipeline DeepShield")
print("=" * 60)

carpetas = [
    ("data/raw/real",              "Imágenes reales (FFHQ)         ← Paso 01"),
    ("data/raw/fake_swap",         "Imágenes falsas (face swap)    ← Paso 02"),
    ("data/Train_D/images",        "Train_D/images/                ← Paso 05"),
    ("data/Train_D/fake_mask",     "Train_D/fake_mask/             ← Paso 04"),
    ("data/Train_D/original_mask", "Train_D/original_mask/         ← Paso 04"),
]

for ruta, nombre in carpetas:
    n, estado = contar(ruta)
    print(f"  {estado}  {nombre:<45} {n:>6} archivos")

print("=" * 60)

# Determinar siguiente paso
n_real  = len(list(Path("data/raw/real").glob("*.png")))         if Path("data/raw/real").exists()              else 0
n_fake  = len(list(Path("data/raw/fake_swap").glob("*.png")))    if Path("data/raw/fake_swap").exists()         else 0
n_fm    = len(list(Path("data/Train_D/fake_mask").glob("*.png")))if Path("data/Train_D/fake_mask").exists()     else 0
n_img   = len(list(Path("data/Train_D/images").glob("*.png")))   if Path("data/Train_D/images").exists()        else 0

print("\nSiguiente paso:")
if n_real == 0:
    print("  → uv run python scripts/01_descargar_ffhq.py")
elif n_fake == 0:
    print("  → ⚠  REQUIERE GPU NVIDIA")
    print("    uv run python scripts/02_generar_swaps.py")
elif n_fm == 0:
    print("  → uv run python scripts/04_generar_mascaras.py")
elif n_img == 0:
    print("  → uv run python scripts/05_ensamblar_dataset.py")
else:
    print("  → ✓ Dataset listo.")
    print("    uv run python src/train.py")