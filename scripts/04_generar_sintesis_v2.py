# scripts/04_generar_sintesis_v2.py
# =============================================================================
# PASO 4 alterno — Síntesis completa mediante Scraping de TPDNE (StyleGAN2)
# =============================================================================
#
# QUÉ HACE:
#   Descarga imágenes de rostros 100% sintéticos desde thispersondoesexist.com.
#   Redimensiona las imágenes a 256x256 para mantener consistencia y genera 
#   sus respectivas máscaras binarias de segmentación (totalmente blancas).
#   Cubre la tipología "síntesis completa" (8.1.1).
#
# POR QUÉ THISPERSONDOESEXIST (StyleGAN2):
#   - Aporta diversidad algorítmica. StyleGAN2 deja huellas forenses distintas 
#     a las de modelos de difusión (Stable Diffusion), como asimetrías en gafas, 
#     fondos incoherentes o artefactos en forma de gota (droplet artifacts).
#
# ESTRATEGIA DE LA MÁSCARA (GROUND-TRUTH):
#   A diferencia del Face Swap o Inpainting, aquí no hay una imagen "original" 
#   con la cual calcular una diferencia absoluta (cv2.absdiff). Dado que el 
#   rostro fue generado desde cero a partir de ruido latente, la imagen entera 
#   es una anomalía. Por lo tanto, la máscara forense para la U-Net será una 
#   matriz de 256x256 llena de píxeles blancos (255), indicando "alteración global".
#
# CONSIDERACIONES TÉCNICAS (Scraping):
#   - TPDNE utiliza cachés agresivos. Si se hacen peticiones muy rápido, 
#     devuelve la misma imagen repetida. Se utiliza un timestamp en la URL y 
#     un retraso (sleep) para forzar la generación de una identidad nueva.
#
# REQUIERE: Conexión a internet. No requiere GPU para este script.
# ENTRADA:  Ninguna (Generación desde cero)
# SALIDA:   data/raw/fake_sintesis_tpdne/  — imágenes sintéticas (StyleGAN2)
#           data/raw/masks_sintesis_tpdne/ — máscaras binarias (bloque blanco)
# =============================================================================

import os
import time
import requests
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_FAKE     = Path("data/raw/fake_sintesis_tpdne")
DIRECTORIO_MASCARAS = Path("data/raw/masks_sintesis_tpdne")
ARCHIVO_LOG         = Path("data/raw/sintesis_tpdne_log.json")

DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)
DIRECTORIO_MASCARAS.mkdir(parents=True, exist_ok=True)

# Parámetros de la prueba
TOTAL_IMAGENES = 50
TIEMPO_ESPERA  = 1.5 # Segundos entre peticiones para evadir el caché de la web
URL_OBJETIVO   = "https://thispersondoesexist.com/image"

# Headers para simular un navegador real y evitar bloqueos (HTTP 403)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
}

# ── Generación de Síntesis y Máscaras ─────────────────────────────────────────
print(f"[1/3] Iniciando scraping de TPDNE (StyleGAN2)...")
print(f"      Objetivo: {TOTAL_IMAGENES} imágenes sintéticas")
print(f"      Espera configurada: {TIEMPO_ESPERA}s (para garantizar rostros únicos)\n")

exitosas = 0
errores  = 0
registro = []

for i in tqdm(range(TOTAL_IMAGENES), desc="Descargando", unit="img"):
    # Nombrar los archivos secuencialmente (ej. tpdne_00000.png)
    nombre_archivo = f"tpdne_{i:05d}.png"
    ruta_destino   = DIRECTORIO_FAKE / nombre_archivo
    ruta_mascara   = DIRECTORIO_MASCARAS / nombre_archivo

    if ruta_destino.exists() and ruta_mascara.exists():
        exitosas += 1
        registro.append({"imagen": nombre_archivo, "estado": "ya_existia"})
        continue

    try:
        # 1. Petición HTTP con evasión de caché
        # Agregamos un query string con el timestamp actual milimétrico para 
        # engañar al servidor y forzarlo a devolver una imagen "fresca".
        timestamp_url = f"{URL_OBJETIVO}?t={int(time.time() * 1000)}"
        respuesta = requests.get(timestamp_url, headers=HEADERS, timeout=10)
        
        if respuesta.status_code == 200:
            # 2. Convertir el buffer de bytes (JPEG) a una matriz NumPy de OpenCV
            imagen_bytes = np.frombuffer(respuesta.content, np.uint8)
            img_cv2 = cv2.imdecode(imagen_bytes, cv2.IMREAD_COLOR)
            
            # 3. Redimensionar de 1024x1024 a 256x256 (Interpolación de alta calidad)
            img_256 = cv2.resize(img_cv2, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # ── EXTRACCIÓN DE LA MÁSCARA FORENSE (SÍNTESIS COMPLETA) ──────────
            # Como la imagen no tiene un origen real, no se usa cv2.absdiff().
            # Se genera programáticamente una matriz de 256x256, de 1 solo canal
            # (escala de grises), y se llena completamente con el valor 255 (Blanco).
            # Esto le enseña a la U-Net que el 100% de los píxeles provienen de 
            # un modelo generativo y no de una cámara real.
            mascara_blanca = np.ones((256, 256), dtype=np.uint8) * 255
            # ──────────────────────────────────────────────────────────────────

            # 4. Guardar resultados en el disco
            cv2.imwrite(str(ruta_destino), img_256)
            cv2.imwrite(str(ruta_mascara), mascara_blanca)
            
            exitosas += 1
            registro.append({"imagen": nombre_archivo, "estado": "exitoso"})
            
            # 5. Pausa obligatoria para evitar baneo de IP o repetición de imágenes
            time.sleep(TIEMPO_ESPERA)
            
        else:
            errores += 1
            registro.append({
                "imagen": nombre_archivo, 
                "estado": f"error_http_{respuesta.status_code}"
            })
            time.sleep(TIEMPO_ESPERA) # Esperar incluso en error para no saturar

    except Exception as e:
        errores += 1
        registro.append({"imagen": nombre_archivo, "estado": f"error: {str(e)}"})
        time.sleep(TIEMPO_ESPERA)

# ── Guardar log y resumen ─────────────────────────────────────────────────────
print(f"\n[2/3] Guardando registros...")
datos_log = {
    "fecha":        datetime.now().isoformat(),
    "fuente":       "thispersondoesexist.com",
    "arquitectura": "StyleGAN2",
    "exitosas":     exitosas,
    "errores":      errores,
    "total_salida": len(list(DIRECTORIO_FAKE.glob("*.png"))),
    "detalle":      registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[3/3] Resumen:")
print(f"  Rostros StyleGAN2 descargados: {exitosas}")
print(f"  Errores de red:                {errores}")
print(f"  Imágenes en fake_sintesis/:    {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"  Máscaras blancas generadas:    {len(list(DIRECTORIO_MASCARAS.glob('*.png')))}")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/05_generar_reenactment.py")