# scripts/04_generar_sintesis_v2.py
# =============================================================================
# PASO 4 alterno — Síntesis completa usando ThisPersonDoesNotExistAPI
# =============================================================================
#
# QUÉ HACE:
#   Descarga imágenes sintéticas utilizando la librería de David-Lor.
#   Redimensiona a 256x256 y genera las máscaras binarias de segmentación 
#   (totalmente blancas).
#
# ESTRATEGIA DE LA MÁSCARA (GROUND-TRUTH):
#   Como el rostro fue generado desde cero, la máscara forense para la U-Net 
#   será una matriz de 256x256 llena de píxeles blancos (255), indicando 
#   "alteración global".
# =============================================================================

import cv2
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from thispersondoesnotexist import get_online_person

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_FAKE     = Path("data/raw/fake_sintesis_tpdne")
DIRECTORIO_MASCARAS = Path("data/raw/masks_sintesis_tpdne")
ARCHIVO_LOG         = Path("data/raw/sintesis_tpdne_log.json")

DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)
DIRECTORIO_MASCARAS.mkdir(parents=True, exist_ok=True)

# Parámetros de la prueba
TOTAL_IMAGENES = 10
TIEMPO_ESPERA  = 3.0 # Segundos entre peticiones para evitar bloqueos

# ── Generación de Síntesis y Máscaras ─────────────────────────────────────────
print(f"[1/3] Iniciando descarga con ThisPersonDoesNotExistAPI...")
print(f"      Objetivo: {TOTAL_IMAGENES} imágenes sintéticas\n")

exitosas = 0
errores  = 0
registro = []

for i in tqdm(range(TOTAL_IMAGENES), desc="Descargando", unit="img"):
    nombre_archivo = f"tpdne_{i:05d}.png"
    ruta_destino   = DIRECTORIO_FAKE / nombre_archivo
    ruta_mascara   = DIRECTORIO_MASCARAS / nombre_archivo

    if ruta_destino.exists() and ruta_mascara.exists():
        exitosas += 1
        registro.append({"imagen": nombre_archivo, "estado": "ya_existia"})
        continue

    try:
        # 1. Petición HTTP usando la API (Devuelve los bytes de la imagen)
        picture_bytes = get_online_person()
        
        # 2. Decodificar bytes a formato OpenCV (Matriz NumPy BGR)
        imagen_np = np.frombuffer(picture_bytes, np.uint8)
        img_cv2   = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            errores += 1
            registro.append({"imagen": nombre_archivo, "estado": "error_decodificacion"})
            time.sleep(TIEMPO_ESPERA)
            continue

        # 3. Redimensionar de 1024x1024 a 256x256
        img_256 = cv2.resize(img_cv2, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        # ── EXTRACCIÓN DE LA MÁSCARA FORENSE (SÍNTESIS COMPLETA) ──────────
        # Matriz de 256x256 llena completamente con el valor 255 (Blanco)
        mascara_blanca = np.ones((256, 256), dtype=np.uint8) * 255
        # ──────────────────────────────────────────────────────────────────

        # 4. Guardar resultados
        cv2.imwrite(str(ruta_destino), img_256)
        cv2.imwrite(str(ruta_mascara), mascara_blanca)
        
        exitosas += 1
        registro.append({"imagen": nombre_archivo, "estado": "exitoso"})
        
        # Pausa de cortesía para el servidor
        time.sleep(TIEMPO_ESPERA)

    except Exception as e:
        errores += 1
        print(f"\n[!] EXCEPCIÓN en {nombre_archivo}: {str(e)}")
        registro.append({"imagen": nombre_archivo, "estado": f"error: {str(e)}"})
        time.sleep(TIEMPO_ESPERA)

# ── Guardar log y resumen ─────────────────────────────────────────────────────
print(f"\n[2/3] Guardando registros...")
datos_log = {
    "fecha":        datetime.now().isoformat(),
    "fuente":       "thispersondoesnotexist API",
    "arquitectura": "StyleGAN2",
    "exitosas":     exitosas,
    "errores":      errores,
    "total_salida": len(list(DIRECTORIO_FAKE.glob("*.png"))),
    "detalle":      registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[3/3] Resumen:")
print(f"  Rostros StyleGAN2 extraídos: {exitosas}")
print(f"  Errores:                     {errores}")
print(f"  Imágenes en fake_sintesis/:  {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"  Máscaras blancas generadas:  {len(list(DIRECTORIO_MASCARAS.glob('*.png')))}")