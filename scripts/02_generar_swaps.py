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
import random
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_REAL = Path("data/raw/real")
DIRECTORIO_FAKE = Path("data/raw/fake_swap")
ARCHIVO_LOG     = Path("data/raw/swap_log.json")
DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)

SEMILLA = 42  # Semilla fija — mismo emparejamiento real/fuente en cualquier dispositivo
random.seed(SEMILLA)

# ── Verificar inswapper_128.onnx ──────────────────────────────────────────────
ruta_swapper = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
if not os.path.exists(ruta_swapper):
    print("ERROR: No se encontró inswapper_128.onnx")
    print(f"Ruta esperada: {ruta_swapper}")
    print("Descárgalo manualmente desde:")
    print("  https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx")
    print(f"Y colócalo en: {ruta_swapper}")
    exit(1)

# ── Inicializar InsightFace ───────────────────────────────────────────────────
# buffalo_l: modelo de detección facial de InsightFace
# providers: CUDA primero, CPU como respaldo si CUDA falla
# ctx_id=0: primera GPU disponible
# det_size=(640,640): resolución de detección — balance entre precisión y velocidad
print("[1/4] Inicializando InsightFace con buffalo_l...")
print("      (Primera ejecución descarga ~280 MB automáticamente)")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ── Cargar modelo de intercambio ──────────────────────────────────────────────
# inswapper opera a 128x128 internamente.
# Esta resolución limitada introduce artefactos de borde detectables (8.7.1).
print("      Cargando inswapper_128.onnx...")
swapper = insightface.model_zoo.get_model(ruta_swapper)

# ── Cargar lista de imágenes ──────────────────────────────────────────────────
imagenes = sorted(DIRECTORIO_REAL.glob("*.png"))
print(f"\n[2/4] Imágenes reales encontradas: {len(imagenes)}")

if len(imagenes) == 0:
    print("ERROR: No hay imágenes en data/raw/real/")
    print("Solución: ejecutar primero  uv run python scripts/01_descargar_ffhq.py")
    exit(1)

# ── Generar intercambios de rostro ────────────────────────────────────────────
# Decisión de diseño — imágenes sin rostro detectable:
#   Se DESCARTAN completamente. No se guarda nada en fake_swap/.
#   Razón: guardar la imagen original como fake contaminaría el dataset
#   porque la sustracción real-fake daría diferencia cero → máscara vacía
#   → el modelo recibiría ruido sin señal forense (8.9.1).
print(f"[3/4] Generando intercambios de rostro...")
print(f"      Semilla fija: {SEMILLA} — reproducible en cualquier dispositivo")
print(f"      Imágenes ya procesadas se saltan automáticamente\n")

exitosos   = 0
sin_rostro = 0
errores    = 0
registro   = []

for ruta_objetivo in tqdm(imagenes, desc="Intercambio de rostro", unit="img"):
    ruta_destino = DIRECTORIO_FAKE / ruta_objetivo.name

    # Reanudar: saltar si ya fue procesada
    if ruta_destino.exists():
        exitosos += 1
        registro.append({"imagen": ruta_objetivo.name, "estado": "ya_existia"})
        continue

    # Elegir imagen fuente aleatoria diferente al objetivo
    candidatos   = [p for p in imagenes if p != ruta_objetivo]
    ruta_fuente  = random.choice(candidatos)

    # Leer imágenes
    img_objetivo = cv2.imread(str(ruta_objetivo))
    img_fuente   = cv2.imread(str(ruta_fuente))

    if img_objetivo is None or img_fuente is None:
        registro.append({"imagen": ruta_objetivo.name, "estado": "error_lectura"})
        errores += 1
        continue

    # Detectar rostros — si no hay rostro en alguna, descartar
    rostros_objetivo = app.get(img_objetivo)
    rostros_fuente   = app.get(img_fuente)

    if not rostros_objetivo:
        registro.append({"imagen": ruta_objetivo.name, "estado": "sin_rostro_objetivo",
                         "fuente": ruta_fuente.name})
        sin_rostro += 1
        continue

    if not rostros_fuente:
        registro.append({"imagen": ruta_objetivo.name, "estado": "sin_rostro_fuente",
                         "fuente": ruta_fuente.name})
        sin_rostro += 1
        continue

    # Aplicar intercambio de rostro
    # paste_back=True: pega el rostro intercambiado sobre la imagen original
    # con blending — esto introduce los artefactos de borde (8.7.1)
    try:
        resultado = img_objetivo.copy()
        resultado = swapper.get(
            resultado,
            rostros_objetivo[0],  # Dónde pegar (geometría del rostro objetivo)
            rostros_fuente[0],    # Qué pegar (identidad del rostro fuente)
            paste_back=True
        )
        cv2.imwrite(str(ruta_destino), resultado)
        exitosos += 1
        registro.append({"imagen": ruta_objetivo.name, "estado": "exitoso",
                          "fuente": ruta_fuente.name})

    except Exception as e:
        registro.append({"imagen": ruta_objetivo.name, "estado": f"error: {str(e)}",
                          "fuente": ruta_fuente.name})
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
    "total_salida":    len(list(DIRECTORIO_FAKE.glob("*.png"))),
    "detalle":         registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[4/4] Resumen:")
print(f"  Intercambios exitosos:      {exitosos}")
print(f"  Sin rostro detectable:      {sin_rostro}  ← descartados, no contaminan el dataset")
print(f"  Errores inesperados:        {errores}  ← descartados, no contaminan el dataset")
print(f"  ────────────────────────────────────")
print(f"  Total en fake_swap/:        {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"  Registro guardado en:       {ARCHIVO_LOG}")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/04_generar_mascaras.py")