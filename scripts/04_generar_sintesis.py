# scripts/04_generar_sintesis.py
# =============================================================================
# PASO 4 — Síntesis completa de rostros con Stable Diffusion text-to-image
# =============================================================================
#
# QUÉ HACE:
#   Genera rostros 100% sintéticos mediante text-to-image con SD.
#   No existe persona real de origen — identidad completamente inventada.
#   Cubre tipología "síntesis completa" de 8.1.1.
#
# POR QUÉ SD text-to-image:
#   - Ya está descargado en caché del paso 03 (no descarga nada nuevo)
#   - Genera artefactos fotométricos y de textura distintos al inpainting
#   - Produce coherencia global alta con inconsistencias locales (8.7.1)
#
# ESTRATEGIA DE ETIQUETADO (8.1.2):
#   Síntesis completa = toda la imagen es manipulada.
#   fake_mask    → rostro completo detectado = blanco
#   original_mask → todo negro (no hay zona auténtica)
#
# REQUIERE: GPU NVIDIA, modelo SD ya descargado del paso 03
# ENTRADA:  data/raw/real/ — solo para saber cuántas imágenes generar
# SALIDA:   data/raw/fake_sintesis/ — rostros completamente sintéticos
# =============================================================================

import argparse
import torch
import cv2
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paso 4 — Síntesis completa de rostros con Stable Diffusion")
    p.add_argument("--data_dir", type=Path, default=Path("data"), help="Directorio raíz de datos")
    return p.parse_args()


_args = _parse_args()

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_REAL = _args.data_dir / "raw/real"
DIRECTORIO_FAKE = _args.data_dir / "raw/fake_sintesis"
ARCHIVO_LOG     = _args.data_dir / "raw/sintesis_log.json"
DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)

SEMILLA        = 42
PASOS_SD       = 25   # Menos pasos que inpainting — síntesis completa necesita menos
random.seed(SEMILLA)

# Prompts variados para diversidad de identidades sintéticas (8.8.2)
PROMPTS = [
    "portrait photo of a person, photorealistic, high detail, natural lighting",
    "close up face portrait, realistic skin texture, photorealistic",
    "human face portrait, different person, natural expression, 8k",
    "realistic face photo, studio lighting, high resolution portrait",
    "photorealistic portrait of a human, natural skin, detailed features",
]

# ── Cargar SD text-to-image ───────────────────────────────────────────────────
# Usa el mismo modelo base de SD ya descargado en el paso 03
# runwayml/stable-diffusion-v1-5 está en caché de HuggingFace
print("[1/4] Cargando Stable Diffusion text-to-image...")
print("      (Usa modelo ya descargado — sin descarga adicional)")

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

pipeline.set_progress_bar_config(disable=True)
print("      Modelo cargado en GPU\n")

# ── Cargar lista de imágenes para saber cuántas generar ──────────────────────
imagenes_reales = sorted(DIRECTORIO_REAL.glob("*.png"))[:109]
total           = len(imagenes_reales)
print(f"[2/4] Generando {total} rostros sintéticos...")
print(f"      Tiempo estimado: ~5 seg/imagen en RTX 4050\n")

# ── Generar imágenes sintéticas ───────────────────────────────────────────────
exitosas = 0
errores  = 0
registro = []

for i, ruta_real in enumerate(tqdm(imagenes_reales, desc="Síntesis", unit="img")):
    # Mismo nombre que la imagen real para mantener trazabilidad
    ruta_destino = DIRECTORIO_FAKE / ruta_real.name

    if ruta_destino.exists():
        exitosas += 1
        registro.append({"imagen": ruta_real.name, "estado": "ya_existia"})
        continue

    try:
        # Prompt y semilla distintos por imagen — diversidad de identidades
        prompt    = random.choice(PROMPTS)
        generador = torch.Generator("cuda").manual_seed(SEMILLA + i)

        # Generar rostro sintético a 512px (resolución nativa de SD)
        resultado = pipeline(
            prompt              = prompt,
            num_inference_steps = PASOS_SD,
            guidance_scale      = 7.5,
            generator           = generador,
            height              = 512,
            width               = 512,
        ).images[0]

        # Redimensionar a 128×128 para consistencia con el resto del dataset
        resultado_final = resultado.resize((256, 256), Image.LANCZOS)
        resultado_final.save(ruta_destino, "PNG")

        exitosas += 1
        registro.append({
            "imagen": ruta_real.name,
            "estado": "exitoso",
            "prompt": prompt,
            "semilla": SEMILLA + i
        })

    except Exception as e:
        errores += 1
        registro.append({"imagen": ruta_real.name,
                         "estado": f"error: {str(e)}"})

# ── Guardar log y resumen ─────────────────────────────────────────────────────
datos_log = {
    "fecha":        datetime.now().isoformat(),
    "modelo":       "runwayml/stable-diffusion-v1-5",
    "pasos_sd":     PASOS_SD,
    "semilla_base": SEMILLA,
    "exitosas":     exitosas,
    "errores":      errores,
    "total_salida": len(list(DIRECTORIO_FAKE.glob("*.png"))),
    "detalle":      registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[4/4] Resumen:")
print(f"  Rostros sintéticos generados: {exitosas}")
print(f"  Errores:                      {errores}")
print(f"  Total en fake_sintesis/:      {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/05_generar_reenactment.py")