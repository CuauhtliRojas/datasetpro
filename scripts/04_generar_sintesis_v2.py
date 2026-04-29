# scripts/04_generar_sintesis_v2.py
# =============================================================================
# PASO 4 alterno — Síntesis completa mediante HuggingFace (StyleGAN2)
# =============================================================================
#
# QUÉ HACE:
#   Descarga imágenes de rostros 100% sintéticos (StyleGAN2) desde un 
#   repositorio curado en HuggingFace, evitando bloqueos de Cloudflare.
#   Redimensiona las imágenes a 256x256 y genera las máscaras binarias.
#
# POR QUÉ ESTE ENFOQUE:
#   Hacer web scraping directo a TPDNE es inestable debido a los escudos
#   anti-bots. Usar el dataset de HuggingFace (nielsr/thispersondoesnotexist)
#   garantiza una ingesta de datos rápida, reproducible y libre de errores HTTP,
#   ideal para alimentar la arquitectura de segmentación.
#
# ESTRATEGIA DE LA MÁSCARA (GROUND-TRUTH):
#   Como el rostro fue generado desde cero, la máscara forense para la U-Net 
#   será una matriz de 256x256 llena de píxeles blancos (255), indicando 
#   "alteración global".
#
# ENTRADA:  Streaming del dataset desde HuggingFace
# SALIDA:   data/raw/fake_sintesis_tpdne/  — imágenes sintéticas (StyleGAN2)
#           data/raw/masks_sintesis_tpdne/ — máscaras binarias (bloque blanco)
# =============================================================================

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_FAKE     = Path("data/raw/fake_sintesis_tpdne")
DIRECTORIO_MASCARAS = Path("data/raw/masks_sintesis_tpdne")
ARCHIVO_LOG         = Path("data/raw/sintesis_tpdne_log.json")

DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)
DIRECTORIO_MASCARAS.mkdir(parents=True, exist_ok=True)

# Parámetros de la prueba
TOTAL_IMAGENES = 10

# ── Generación de Síntesis y Máscaras ─────────────────────────────────────────
print(f"[1/3] Conectando al dataset de StyleGAN2 en HuggingFace...")
print(f"      Objetivo: {TOTAL_IMAGENES} imágenes sintéticas\n")

exitosas = 0
errores  = 0
registro = []

try:
    # Cargamos el dataset en modo 'streaming' para no descargar los GBs completos,
    # solo tomamos los que necesitamos sobre la marcha.
    dataset = load_dataset("nielsr/thispersondoesnotexist", split="train", streaming=True)
    
    # Tomar solo la cantidad necesaria
    iterator = iter(dataset.take(TOTAL_IMAGENES))
    
    for i in tqdm(range(TOTAL_IMAGENES), desc="Procesando", unit="img"):
        nombre_archivo = f"tpdne_{i:05d}.png"
        ruta_destino   = DIRECTORIO_FAKE / nombre_archivo
        ruta_mascara   = DIRECTORIO_MASCARAS / nombre_archivo

        if ruta_destino.exists() and ruta_mascara.exists():
            exitosas += 1
            registro.append({"imagen": nombre_archivo, "estado": "ya_existia"})
            next(iterator) # Avanzar el iterador para no repetir
            continue

        try:
            # Obtener la imagen PIL del dataset
            item = next(iterator)
            imagen_pil = item['image']
            
            # Convertir PIL (RGB) a OpenCV (BGR)
            img_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
            
            # Redimensionar de 1024x1024 a 256x256 (Interpolación Lanczos)
            img_256 = cv2.resize(img_cv2, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # ── EXTRACCIÓN DE LA MÁSCARA FORENSE (SÍNTESIS COMPLETA) ──────────
            mascara_blanca = np.ones((256, 256), dtype=np.uint8) * 255
            # ──────────────────────────────────────────────────────────────────

            # Guardar en disco
            cv2.imwrite(str(ruta_destino), img_256)
            cv2.imwrite(str(ruta_mascara), mascara_blanca)
            
            exitosas += 1
            registro.append({"imagen": nombre_archivo, "estado": "exitoso"})

        except StopIteration:
            print("\n[!] Se agotaron las imágenes del dataset en streaming.")
            break
        except Exception as e:
            errores += 1
            registro.append({"imagen": nombre_archivo, "estado": f"error_procesamiento: {str(e)}"})

except Exception as e:
    print(f"\n[!] Error fatal al conectar con HuggingFace: {e}")

# ── Guardar log y resumen ─────────────────────────────────────────────────────
print(f"\n[2/3] Guardando registros...")
datos_log = {
    "fecha":        datetime.now().isoformat(),
    "fuente":       "HuggingFace (nielsr/thispersondoesnotexist)",
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
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/05_generar_reenactment.py")