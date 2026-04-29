# scripts/imprimir_tabla.py
import sys
from pathlib import Path
from PIL import Image, ImageDraw

# ── 1. Definir Directorios ────────────────────────────────────────────────────
DIR_REAL      = Path('data/raw/real')
DIR_SWAP      = Path('data/raw/fake_swap')
DIR_SWAP_M    = Path('data/raw/masks_fake_swap')
DIR_INPAINT   = Path('data/raw/fake_inpainting')
DIR_INPAINT_M = Path('data/raw/masks_inpainting')
DIR_REENACT   = Path('data/raw/fake_reenactment')
DIR_REENACT_M = Path('data/raw/masks_reenactment')

DIRECTORIOS = [
    DIR_REAL, 
    DIR_SWAP, DIR_SWAP_M, 
    DIR_INPAINT, DIR_INPAINT_M, 
    DIR_REENACT, DIR_REENACT_M
]

COLUMNAS = [
    'REAL', 
    'SWAP', 'MASK_SWAP', 
    'INPAINT', 'MASK_INPAINT', 
    'REENACT', 'MASK_REENACT'
]

# ── 2. Lógica de Intersección (Encontrar muestras completas) ──────────────────
print("Buscando imágenes que existan en todas las carpetas (intersección)...")

# Tomamos como base la carpeta de Swap porque suele ser la que más descarta rostros
nombres_base = [p.name for p in DIR_SWAP.glob('*.png')]
muestras = []

for nombre in sorted(nombres_base):
    # Verificar si el archivo existe en todos los directorios requeridos
    if all((d / nombre).exists() for d in DIRECTORIOS):
        muestras.append(nombre)
    
    # Detenernos cuando tengamos las 5 necesarias
    if len(muestras) == 5:
        break

if not muestras:
    print("ERROR: No se encontró ninguna imagen que haya pasado por todos los procesos.")
    sys.exit(1)

if len(muestras) < 5:
    print(f"Advertencia: Solo se encontraron {len(muestras)} muestras completas.")

print(f"Muestras seleccionadas: {muestras}")

# ── 3. Configuración de la Grilla (Grid) ──────────────────────────────────────
TAM = 200  # Tamaño en píxeles de cada imagen (200x200)
SEP = 4    # Separación entre columnas/filas

n_cols  = len(COLUMNAS)
n_filas = len(muestras)

ancho_total = n_cols * TAM + (n_cols - 1) * SEP
alto_total  = n_filas * TAM + 30 + ((n_filas - 1) * SEP) # +30px para las etiquetas

# Fondo gris oscuro
grid = Image.new('RGB', (ancho_total, alto_total), (30, 30, 30))
draw = ImageDraw.Draw(grid)

# Paleta de colores para los encabezados
colores_col = [
    (80, 80, 80),    # REAL (Gris)
    (180, 60, 60),   # SWAP (Rojo)
    (120, 40, 40),   # MASK SWAP (Rojo oscuro)
    (60, 120, 180),  # INPAINT (Azul)
    (40, 80, 120),   # MASK INPAINT (Azul oscuro)
    (180, 120, 40),  # REENACT (Naranja)
    (120, 80, 25)    # MASK REENACT (Naranja oscuro)
]

# ── 4. Dibujar Encabezados ────────────────────────────────────────────────────
for c, (nombre, color) in enumerate(zip(COLUMNAS, colores_col)):
    x = c * (TAM + SEP)
    draw.rectangle([x, 0, x + TAM, 28], fill=color)
    draw.text((x + 6, 8), nombre, fill=(255, 255, 255))

# ── 5. Ensamblar Imágenes ─────────────────────────────────────────────────────
for f, nombre in enumerate(muestras):
    for c, directorio in enumerate(DIRECTORIOS):
        ruta = directorio / nombre
        
        # Abrir y redimensionar (usando Resampling para compatibilidad Pillow 10+)
        img = Image.open(ruta).convert('RGB')
        img = img.resize((TAM, TAM), Image.Resampling.NEAREST)

        x = c * (TAM + SEP)
        y = 30 + f * (TAM + SEP)
        grid.paste(img, (x, y))

# ── 6. Guardar ────────────────────────────────────────────────────────────────
ruta_salida = 'data/comparacion.png'
grid.save(ruta_salida)
print(f"\n¡Éxito! Grilla guardada en: {ruta_salida}")
print("Abre el archivo en tu explorador de archivos para inspeccionar las máscaras.")