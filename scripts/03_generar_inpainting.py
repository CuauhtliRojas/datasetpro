# scripts/03_generar_inpainting.py
# =============================================================================
# PASO 3 DEL PIPELINE — Generación de fakes por edición local (inpainting)
# =============================================================================
#
# QUÉ HACE:
#   Usa Stable Diffusion Inpainting para editar zonas específicas del rostro
#   (boca, ojos, zona periocular) reemplazándolas con contenido sintético.
#   Cubre la tipología "edición local" de §8.1.1.
#
# POR QUÉ SD INPAINTING (§8.1.1, §8.7.1):
#   - Edita regiones acotadas del rostro sin reemplazar la identidad completa
#   - Introduce inconsistencias fotométricas locales detectables (§8.7.1)
#   - Genera discontinuidades en bordes de la región editada
#   - Complementa el reemplazo de rostro del script 02 con artefactos distintos
#
# ARTEFACTOS QUE INTRODUCE:
#   - Inconsistencias fotométricas en la zona editada vs contexto original
#   - Bordes de fusión entre región generada y región auténtica
#   - Diferencias de textura de piel en zona inpainted
#
# REQUIERE: GPU NVIDIA + cuenta HuggingFace con token configurado
# ENTRADA:  data/raw/real/              — imágenes reales base
# SALIDA:   data/raw/fake_inpainting/   — imágenes con edición local
#           data/raw/inpainting_log.json
# =============================================================================

import torch
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_REAL  = Path("data/raw/real")
DIRECTORIO_FAKE  = Path("data/raw/fake_inpainting")
ARCHIVO_LOG      = Path("data/raw/inpainting_log.json")
DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)

# Regiones faciales a editar — coordenadas relativas (0.0 a 1.0)
# Formato: (x_inicio, y_inicio, x_fin, y_fin)
# Se eligen aleatoriamente para variar la zona editada por imagen
REGIONES_FACIALES = {
    "boca":          (0.25, 0.60, 0.75, 0.85),   # Zona labial (§8.3.1)
    "ojo_izquierdo": (0.10, 0.25, 0.45, 0.50),   # Región periocular izq
    "ojo_derecho":   (0.55, 0.25, 0.90, 0.50),   # Región periocular der
    "nariz":         (0.30, 0.40, 0.70, 0.65),   # Zona nasal
}

# Prompts para guiar la generación — variedad para evitar sesgo de método (§8.8.2)
PROMPTS = [
    "realistic human face, natural skin texture, photorealistic",
    "person face with different features, photorealistic, 8k",
    "realistic facial features, natural lighting, high detail",
    "human face, different person, photorealistic skin",
]

SEMILLA   = 42
PASOS_SD  = 30   # Pasos de difusión — balance calidad/velocidad en RTX 4050
GUIA      = 7.5  # Guidance scale

# ── Cargar modelo SD Inpainting ───────────────────────────────────────────────
# runwayml/stable-diffusion-inpainting: modelo optimizado para inpainting
# Descarga ~5GB en primera ejecución desde HuggingFace
print("[1/4] Cargando Stable Diffusion Inpainting...")
print("      (Primera ejecución descarga ~5GB desde HuggingFace)")

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,    # float16 para caber en 6GB VRAM
    safety_checker=None,          # Desactivar para rostros
    requires_safety_checker=False
).to("cuda")

pipeline.set_progress_bar_config(disable=True)  # Silenciar barra interna de SD

print("      Modelo cargado correctamente en GPU")

# ── Funciones auxiliares ──────────────────────────────────────────────────────
def crear_mascara_region(ancho, alto, region_rel):
    """
    Crea máscara binaria PIL para la región facial a editar.
    region_rel: tupla (x0, y0, x1, y1) en coordenadas relativas 0.0-1.0
    """
    mascara = Image.new("RGB", (ancho, alto), "black")
    x0 = int(region_rel[0] * ancho)
    y0 = int(region_rel[1] * alto)
    x1 = int(region_rel[2] * ancho)
    y1 = int(region_rel[3] * alto)

    # Dibujar región blanca (zona a inpaint)
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(mascara)
    draw.rectangle([x0, y0, x1, y1], fill="white")
    return mascara, (x0, y0, x1, y1)

# ── Cargar imágenes ───────────────────────────────────────────────────────────
import random
random.seed(SEMILLA)
np.random.seed(SEMILLA)

imagenes = sorted(DIRECTORIO_REAL.glob("*.png"))
print(f"\n[2/4] Imágenes reales encontradas: {len(imagenes)}")

if len(imagenes) == 0:
    print("ERROR: No hay imágenes en data/raw/real/")
    print("Solución: ejecutar primero  uv run python scripts/01_descargar_ffhq.py")
    exit(1)

# ── Generar inpaintings ───────────────────────────────────────────────────────
print(f"[3/4] Generando ediciones locales con SD Inpainting...")
print(f"      Región editada varía por imagen para diversidad de dataset")
print(f"      Tiempo estimado: ~20 segundos por imagen en RTX 4050\n")

exitosas   = 0
errores    = 0
registro   = []

nombres_regiones = list(REGIONES_FACIALES.keys())
nombres_prompts  = PROMPTS

generador = torch.Generator("cuda").manual_seed(SEMILLA)

for i, ruta_real in enumerate(tqdm(imagenes, desc="Inpainting", unit="img")):
    ruta_destino = DIRECTORIO_FAKE / ruta_real.name

    if ruta_destino.exists():
        exitosas += 1
        registro.append({"imagen": ruta_real.name, "estado": "ya_existia"})
        continue

    try:
        # Cargar imagen y redimensionar a 512px (requerimiento de SD)
        img_pil = Image.open(ruta_real).convert("RGB").resize((512, 512))
        ancho, alto = img_pil.size

        # Elegir región y prompt aleatoriamente
        nombre_region = random.choice(nombres_regiones)
        region_rel    = REGIONES_FACIALES[nombre_region]
        prompt        = random.choice(nombres_prompts)

        # Crear máscara de la región a editar
        mascara_pil, coords = crear_mascara_region(ancho, alto, region_rel)

        # Ejecutar inpainting
        resultado = pipeline(
            prompt          = prompt,
            image           = img_pil,
            mask_image      = mascara_pil,
            num_inference_steps = PASOS_SD,
            guidance_scale  = GUIA,
            generator       = generador,
        ).images[0]

        # Redimensionar de vuelta al tamaño original (128x128)
        img_original = Image.open(ruta_real).convert("RGB")
        w_orig, h_orig = img_original.size
        resultado_final = resultado.resize((w_orig, h_orig), Image.LANCZOS)
        resultado_final.save(ruta_destino, "PNG")

        exitosas += 1
        registro.append({
            "imagen":  ruta_real.name,
            "estado":  "exitoso",
            "region":  nombre_region,
            "coords":  coords,
            "prompt":  prompt
        })

    except Exception as e:
        errores += 1
        registro.append({"imagen": ruta_real.name, "estado": f"error: {str(e)}"})
        continue

# ── Guardar log y resumen ─────────────────────────────────────────────────────
datos_log = {
    "fecha":        datetime.now().isoformat(),
    "semilla":      SEMILLA,
    "modelo":       "runwayml/stable-diffusion-inpainting",
    "pasos_sd":     PASOS_SD,
    "exitosas":     exitosas,
    "errores":      errores,
    "total_salida": len(list(DIRECTORIO_FAKE.glob("*.png"))),
    "detalle":      registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[4/4] Resumen:")
print(f"  Inpaintings exitosos:  {exitosas}")
print(f"  Errores:               {errores}")
print(f"  Total en fake_inpainting/: {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"  Registro guardado en:  {ARCHIVO_LOG}")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/07_generar_sintesis.py")