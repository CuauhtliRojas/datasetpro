# scripts/02_generar_swaps.py
# =============================================================================
# PASO 2 DEL PIPELINE — Generación de imágenes falsas con intercambio de rostro
# =============================================================================
#
# QUÉ HACE:
#   Por cada imagen real en data/raw/real/, genera una imagen falsa donde
#   el rostro original fue reemplazado por el rostro de otra persona del dataset.
#   Esto cubre la tipología "reemplazo de rostro" definida en 8.1.1.
#
# POR QUÉ inswapper_128 (8.1.1, 8.7.1):
#   - Cubre reemplazo de identidad, tipología prioritaria del documento
#   - Introduce artefactos forenses detectables: discontinuidades en bordes
#     de mezcla, halos, inconsistencias fotométricas (8.7.1)
#   - Ejecutable localmente sin API externa (modo offline 8.11.1)
#   - Automatizable por script sin interfaz gráfica
#
# ARTEFACTOS QUE INTRODUCE (señales que U-Net aprenderá a detectar):
#   - Bordes de mezcla entre rostro sintético y cuello/fondo real
#   - Halos de transición en el contorno facial
#   - Inconsistencias de iluminación entre rostro pegado y escena original
#
# REQUISITO MANUAL — inswapper_128.onnx:
#   Descargar desde HuggingFace (~500 MB) y colocar en:
#   C:\Users\TU_USUARIO\.insightface\models\inswapper_128.onnx
#   URL: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
#
# REQUIERE: GPU NVIDIA con CUDA
# ENTRADA:  data/raw/real/       — imágenes reales (PNG numerados)
# SALIDA:   data/raw/fake_swap/  — imágenes con rostro intercambiado
#           data/raw/swap_log.json — registro detallado de resultados
#
# CÓMO REANUDAR SI SE INTERRUMPE:
#   Vuelve a ejecutar el script. Las imágenes ya procesadas se saltan.
# =============================================================================

import cv2
import numpy as np
import random
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_REAL     = Path("data/raw/real")
DIRECTORIO_FAKE     = Path("data/raw/fake_swap")
DIRECTORIO_MASCARAS = Path("data/raw/masks_fake_swap") # NUEVO: Directorio para las máscaras
ARCHIVO_LOG         = Path("data/raw/swap_log.json")

DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)
DIRECTORIO_MASCARAS.mkdir(parents=True, exist_ok=True) # Crear directorio

SEMILLA = 42
random.seed(SEMILLA)

# ── Verificar inswapper_128.onnx ──────────────────────────────────────────────
ruta_swapper = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
if not os.path.exists(ruta_swapper):
    print("ERROR: No se encontró inswapper_128.onnx")
    exit(1)

# ── Inicializar InsightFace ───────────────────────────────────────────────────
print("[1/4] Inicializando InsightFace con buffalo_l...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

print("      Cargando inswapper_128.onnx...")
swapper = insightface.model_zoo.get_model(ruta_swapper)

# ── Cargar lista de imágenes ──────────────────────────────────────────────────
imagenes = sorted(DIRECTORIO_REAL.glob("*.png"))
print(f"\n[2/4] Imágenes reales encontradas: {len(imagenes)}")

if len(imagenes) == 0:
    print("ERROR: No hay imágenes en data/raw/real/")
    exit(1)

# ── Generar intercambios y máscaras ───────────────────────────────────────────
print(f"[3/4] Generando intercambios de rostro y máscaras binarias...")

exitosos   = 0
sin_rostro = 0
errores    = 0
registro   = []

for ruta_objetivo in tqdm(imagenes, desc="Procesando", unit="img"):
    ruta_destino = DIRECTORIO_FAKE / ruta_objetivo.name
    ruta_mascara = DIRECTORIO_MASCARAS / ruta_objetivo.name

    if ruta_destino.exists() and ruta_mascara.exists():
        exitosos += 1
        registro.append({"imagen": ruta_objetivo.name, "estado": "ya_existia"})
        continue

    candidatos  = [p for p in imagenes if p != ruta_objetivo]
    ruta_fuente = random.choice(candidatos)

    img_objetivo = cv2.imread(str(ruta_objetivo))
    img_fuente   = cv2.imread(str(ruta_fuente))

    if img_objetivo is None or img_fuente is None:
        registro.append({"imagen": ruta_objetivo.name, "estado": "error_lectura"})
        errores += 1
        continue

    rostros_objetivo = app.get(img_objetivo)
    rostros_fuente   = app.get(img_fuente)

    if not rostros_objetivo:
        registro.append({"imagen": ruta_objetivo.name, "estado": "sin_rostro_objetivo"})
        sin_rostro += 1
        continue

    if not rostros_fuente:
        registro.append({"imagen": ruta_objetivo.name, "estado": "sin_rostro_fuente"})
        sin_rostro += 1
        continue

    try:
        resultado = img_objetivo.copy()
        resultado = swapper.get(
            resultado,
            rostros_objetivo[0],
            rostros_fuente[0],
            paste_back=True
        )
        
        # ── EXTRACCIÓN DE LA MÁSCARA BINARIA ──────────────────────────────
        # 1. Calcular la diferencia absoluta entre la original y el swap
        diferencia = cv2.absdiff(img_objetivo, resultado)
        
        # 2. Convertir a escala de grises
        diff_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
        
        # 3. Binarizar: Cualquier pixel con una diferencia > 5 se marca como blanco (255)
        # Usamos 5 en lugar de 0 para ignorar ruidos mínimos de compresión
        _, mascara_binaria = cv2.threshold(diff_gris, 5, 255, cv2.THRESH_BINARY)
        
        # 4. Operaciones morfológicas para limpiar ruido (artefactos sueltos) y rellenar huecos
        kernel = np.ones((5,5), np.uint8)
        mascara_limpia = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel) # Elimina ruido externo
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel) # Rellena huecos internos
        
        # Guardar resultados
        cv2.imwrite(str(ruta_destino), resultado)
        cv2.imwrite(str(ruta_mascara), mascara_limpia)
        # ──────────────────────────────────────────────────────────────────

        exitosos += 1
        registro.append({"imagen": ruta_objetivo.name, "estado": "exitoso", "fuente": ruta_fuente.name})

    except Exception as e:
        registro.append({"imagen": ruta_objetivo.name, "estado": f"error: {str(e)}"})
        errores += 1
        continue

# ── Guardar registro y mostrar resumen ────────────────────────────────────────
datos_log = {
    "fecha":           datetime.now().isoformat(),
    "semilla":         SEMILLA,
    "total_entrada":   len(imagenes),
    "exitosos":        exitosos,
    "sin_rostro":      sin_rostro,
    "errores":         errores,
    "detalle":         registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[4/4] Resumen:")
print(f"  Intercambios exitosos:      {exitosos}")
print(f"  Total en fake_swap/:        {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"  Total máscaras generadas:   {len(list(DIRECTORIO_MASCARAS.glob('*.png')))}")
print(f"  Registro guardado en:       {ARCHIVO_LOG}")