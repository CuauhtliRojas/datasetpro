# scripts/06_verify_dataset.py
"""
Verifica el estado actual del proyecto sin necesitar GPU.
Puedes correr esto en cualquier momento y en cualquier dispositivo.
"""

from pathlib import Path

def contar(carpeta, extension="*.png"):
    p = Path(carpeta)
    if not p.exists():
        return 0, "❌ carpeta no existe"
    n = len(list(p.glob(extension)))
    return n, "✓" if n > 0 else "⚠ vacía"

print("=" * 50)
print("ESTADO DEL DATASET — DeepShield Pipeline")
print("=" * 50)

carpetas = [
    ("data/raw/real",              "Imágenes reales (FFHQ)"),
    ("data/raw/fake_swap",         "Fakes generados (SimSwap)"),
    ("data/Train_D/images",        "Train_D/images"),
    ("data/Train_D/fake_mask",     "Train_D/fake_mask"),
    ("data/Train_D/original_mask", "Train_D/original_mask"),
]

for ruta, nombre in carpetas:
    n, estado in contar(ruta):
    print(f"  {estado}  {nombre:<35} {n:>6} archivos")

print("=" * 50)
print("\nPróximo paso pendiente:")

real_n   = len(list(Path("data/raw/real").glob("*.png")))       if Path("data/raw/real").exists()      else 0
fake_n   = len(list(Path("data/raw/fake_swap").glob("*.png")))  if Path("data/raw/fake_swap").exists() else 0
train_n  = len(list(Path("data/Train_D/images").glob("*.png"))) if Path("data/Train_D/images").exists() else 0

if real_n == 0:
    print("  → Ejecutar 01_download_ffhq.py")
elif fake_n == 0:
    print("  → ⚠ REQUIERE LAPTOP CON GPU")
    print("    Ejecutar 02_generate_swaps.py en Laptop RTX 4050")
elif train_n == 0:
    print("  → Ejecutar 04_generate_masks.py")
    print("  → Ejecutar 05_assemble_dataset.py")
else:
    print("  → Dataset listo. Ejecutar src/train_.py en Laptop")