# scripts/02_generate_swaps.py
# =============================================================================
# PASO 2 DEL PIPELINE — Generación de imágenes fake con face swap
# =============================================================================
#
# Dcouemtno: """
#
# Que hace:
#   Por cada imagen real en data/raw/real/, genera una imagen fake donde
#   el rostro original fue reemplazado por el rostro de otra persona.
#   Este proceso cubre la tipología "reemplazo de rostro" definida en
#   8.1.1 del documento.
#
# POR QUÉ inswapper_128 (justificación 8.1.1, 8.7.1):
#   - Cubre reemplazo de identidad, tipología prioritaria del documento
#   - Introduce los artefactos forenses requeridos: discontinuidades en
#     bordes de mezcla, halos, inconsistencias fotométricas (8.7.1)
#   - Ejecutable localmente sin API externa (coherente con 8.11.1)
#   - Automatizable por script sin GUI (requerimiento del pipeline)
#
# ARrtefactos que introduce (lo que el modelo U-Net aprenderá a detectar):
#   - Bordes de mezcla entre rostro sintético y cuello/fondo real
#   - Halos de transición en contorno facial
#   - Inconsistencias de iluminación entre rostro pegado y escena
#   - Diferencias de textura de piel entre región swapped y original
#
# Requiere: GPU NVIDIA con CUDA (ejecutar en Laptop RTX 4050)
#
# ENTRADA:  data/raw/real/       — 1,000 PNG de FFHQ (imágenes auténticas)
# SALIDA:   data/raw/fake_swap/  — 1,000 PNG con rostro intercambiado
#           data/raw/swap_log.txt — registro de resultados por imagen
#
# CÓMO VERIFICAR EL RESULTADO:
#   1. Contar archivos: debe haber exactamente N archivos en fake_swap/
#      donde N <= cantidad de imágenes reales con rostro detectable
#   2. Inspección visual: abrir 5-10 pares (real + fake) lado a lado
#      y confirmar que el rostro cambió pero el fondo es idéntico
#   3. Correr script 06_verify_dataset.py para conteo automático
#
# CÓMO REANUDAR SI SE INTERRUMPE:
#   Vuelve a correr el script. Las imágenes ya procesadas se saltan
#   automáticamente (verifica dest_path.exists()).
# =============================================================================

import cv2
import random
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

REAL_DIR = Path("data/raw/real")
FAKE_DIR = Path("data/raw/fake_swap")
LOG_FILE = Path("data/raw/swap_log.json")

FAKE_DIR.mkdir(parents=True, exist_ok=True)

# Semilla fija para reproducibilidad — misma semilla = mismo par real/fuente
# en cualquier dispositivo (8.8.3 del documento)
SEED = 42
random.seed(SEED)

# =============================================================================
# PASO 1 — Inicializar InsightFace con buffalo_l
# =============================================================================
# buffalo_l es el modelo de detección facial de InsightFace.
# Descarga automática en primera ejecución (~500 MB en ~/.insightface/).
# providers: CUDA primero, CPU como fallback si CUDA falla al inicializar.
# ctx_id=0 apunta a la primera GPU disponible.
# det_size=(640,640): resolución de detección — más alta = más precisa
# pero más lenta. 640 es el balance recomendado por InsightFace.
# =============================================================================

print("[1/4] Inicializando InsightFace...")
print("      (Primera ejecución descarga ~500MB en ~/.insightface/)")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    # CUDA primero — si falla, usa CPU automáticamente
)
app.prepare(ctx_id=0, det_size=(640, 640))

# =============================================================================
# PASO 2 — Cargar inswapper_128
# =============================================================================
# inswapper_128 es el modelo de intercambio de rostros.
# Trabaja a 128x128 internamente — esta resolución limitada es precisamente
# la que introduce los artefactos de borde detectables por U-Net (8.7.1).
# La discrepancia entre la resolución del swap (128px) y la imagen original
# (128px en nuestro caso, 1024px en FFHQ original) genera las huellas
# forenses que el segmentador aprenderá a localizar.
# =============================================================================

print("      Cargando inswapper_128.onnx...")
swapper = insightface.model_zoo.get_model(
    "inswapper_128.onnx",
    download=True,
    download_zip=True
)

# =============================================================================
# PASO 3 — Cargar lista de imágenes reales
# =============================================================================

images = sorted(REAL_DIR.glob("*.png"))
print(f"\n[2/4] Imágenes reales encontradas: {len(images)}")

if len(images) == 0:
    print("ERROR: No hay imágenes en data/raw/real/")
    print("Solución: ejecutar primero  uv run python scripts/01_download_ffhq.py")
    exit(1)

# =============================================================================
# PASO 4 — Generar face swaps
# =============================================================================
# Estrategia de emparejamiento:
#   Cada imagen target recibe un rostro fuente ALEATORIO diferente a ella
#   misma. La semilla fija garantiza que el emparejamiento sea reproducible.
#
# Decisión de diseño — qué hacer cuando no hay rostro detectable:
#   SALTAMOS la imagen completamente (no la guardamos en fake_swap/).
#   Razón: guardar la imagen original en fake_swap/ contaminaría el dataset
#   porque la sustracción real-fake daría diferencia cero y la máscara
#   estaría vacía — el modelo recibiría ruido sin señal forense (8.9.1).
#
# Decisión de diseño — qué hacer en caso de error inesperado:
#   También saltamos. Mismo razonamiento — es preferible tener menos
#   muestras limpias que más muestras contaminadas.
# =============================================================================

print(f"[3/4] Generando face swaps...")
print(f"      Semilla: {SEED} (reproducible)")
print(f"      Imágenes ya procesadas se saltan automáticamente\n")

exitosos   = 0
sin_rostro = 0
errores    = 0
log        = []   # Registro detallado por imagen

for target_path in tqdm(images, desc="SwapFace", unit="img"):
    dest_path = FAKE_DIR / target_path.name

    # ── Reanudar: saltar si ya fue procesada ────
    if dest_path.exists():
        exitosos += 1
        log.append({"img": target_path.name, "status": "skip_exists"})
        continue

    # ── Elegir fuente aleatoria diferente al target ────
    candidates  = [p for p in images if p != target_path]
    source_path = random.choice(candidates)

    # ── Leer imágenes ────
    target_img = cv2.imread(str(target_path))
    source_img = cv2.imread(str(source_path))

    if target_img is None or source_img is None:
        log.append({"img": target_path.name, "status": "error_read"})
        errores += 1
        continue

    # ── Detectar rostros ───
    # app.get() devuelve lista de rostros detectados en la imagen.
    # Usamos solo el primero (índice 0) — el de mayor confianza.
    # Si no detecta rostro en target O en source, saltamos AMBOS.
    target_faces = app.get(target_img)
    source_faces = app.get(source_img)

    if not target_faces:
        log.append({"img": target_path.name, "status": "no_face_target",
                    "source": source_path.name})
        sin_rostro += 1
        continue   # NO guardamos nada — imagen descartada

    if not source_faces:
        log.append({"img": target_path.name, "status": "no_face_source",
                    "source": source_path.name})
        sin_rostro += 1
        continue   # NO guardamos nada — imagen descartada

    # ── Aplicar swap ────
    # paste_back=True: el rostro swapped se pega de vuelta sobre la imagen
    # original con blending — esto es lo que introduce los artefactos de
    # borde en la zona de mezcla (8.7.1).
    try:
        result = target_img.copy()
        result = swapper.get(
            result,
            target_faces[0],   # Dónde pegar (geometría del rostro target)
            source_faces[0],   # Qué pegar (identidad del rostro fuente)
            paste_back=True
        )
        cv2.imwrite(str(dest_path), result)
        exitosos += 1
        log.append({"img": target_path.name, "status": "ok",
                    "source": source_path.name})

    except Exception as e:
        # Error inesperado — saltamos sin guardar nada
        log.append({"img": target_path.name, "status": f"error: {str(e)}",
                    "source": source_path.name})
        errores += 1
        continue   # NO guardamos — imagen descartada

# =============================================================================
# PASO 5 — Guardar log y mostrar resumen
# =============================================================================

log_data = {
    "timestamp":   datetime.now().isoformat(),
    "seed":        SEED,
    "total_input": len(images),
    "exitosos":    exitosos,
    "sin_rostro":  sin_rostro,
    "errores":     errores,
    "total_output": len(list(FAKE_DIR.glob("*.png"))),
    "detalle":     log
}

with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=2, ensure_ascii=False)

print(f"\n[4/4] Resumen:")
print(f"  Swaps exitosos:         {exitosos}")
print(f"  Sin rostro detectable:  {sin_rostro}  ← descartadas (no contaminan dataset)")
print(f"  Errores inesperados:    {errores}  ← descartadas (no contaminan dataset)")
print(f"  ─────────────────────────────────")
print(f"  Total en fake_swap/:    {len(list(FAKE_DIR.glob('*.png')))}")
print(f"  Log guardado en:        {LOG_FILE}")
print(f"\n  SIGUIENTE PASO:")
print(f"  uv run python scripts/04_generate_masks.py")