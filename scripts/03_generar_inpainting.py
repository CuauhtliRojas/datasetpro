# scripts/03_generar_inpainting.py
# =============================================================================
# PASO 3 — Generación de fakes locales y extracción de máscaras
# =============================================================================
#
# QUÉ HACE:
#   1. Usa Stable Diffusion Inpainting para editar zonas del rostro.
#   2. Calcula matemáticamente la diferencia entre la imagen original y la 
#      generada para extraer la máscara binaria del área REALMENTE alterada.
#
# POR QUÉ SD INPAINTING (8.1.1, 8.7.1):
#   - Edita regiones acotadas del rostro sin reemplazar la identidad completa.
#   - Introduce inconsistencias fotométricas locales y artefactos del VAE.
#
# POR QUÉ MÁSCARAS POR DIFERENCIA (Ground-Truth):
#   - Aunque le damos una caja blanca a SD para que pinte ahí, el proceso de
#     difusión y el decodificador VAE difuminan bordes y alteran píxeles
#     aledaños. Calcular cv2.absdiff nos da la huella forense exacta de
#     lo que se modificó, siendo una etiqueta perfecta para la U-Net.
#
# REQUIERE: GPU NVIDIA + cuenta HuggingFace con token configurado
# ENTRADA:  data/raw/real/             — imágenes reales base
# SALIDA:   data/raw/fake_inpainting/  — imágenes con edición local
#           data/raw/masks_inpainting/ — máscaras binarias de la manipulación
#           data/raw/inpainting_log.json
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
from diffusers import StableDiffusionInpaintPipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paso 3 — Inpainting facial con Stable Diffusion")
    p.add_argument("--data_dir", type=Path, default=Path("data"), help="Directorio raíz de datos")
    return p.parse_args()


_args = _parse_args()

# ── Configuración ─────────────────────────────────────────────────────────────
DIRECTORIO_REAL      = _args.data_dir / "raw/real"
DIRECTORIO_FAKE      = _args.data_dir / "raw/fake_inpainting"
DIRECTORIO_MASCARAS  = _args.data_dir / "raw/masks_inpainting"
ARCHIVO_LOG          = _args.data_dir / "raw/inpainting_log.json"

DIRECTORIO_FAKE.mkdir(parents=True, exist_ok=True)
DIRECTORIO_MASCARAS.mkdir(parents=True, exist_ok=True)

# Regiones faciales a editar — coordenadas relativas (0.0 a 1.0)
REGIONES_FACIALES = {
    "boca":          (0.25, 0.60, 0.75, 0.85),
    "ojo_izquierdo": (0.10, 0.25, 0.45, 0.50),
    "ojo_derecho":   (0.55, 0.25, 0.90, 0.50),
    "nariz":         (0.30, 0.40, 0.70, 0.65),
}

PROMPTS = [
    "realistic human face, natural skin texture, photorealistic",
    "person face with different features, photorealistic, 8k",
    "realistic facial features, natural lighting, high detail",
    "human face, different person, photorealistic skin",
]

SEMILLA  = 42
PASOS_SD = 30
GUIA     = 7.5

# Fijar semillas para reproducibilidad del experimento
random.seed(SEMILLA)
np.random.seed(SEMILLA)

# ── Cargar modelo SD Inpainting ───────────────────────────────────────────────
print("[1/4] Cargando Stable Diffusion Inpainting...")
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

pipeline.set_progress_bar_config(disable=True)
print("      Modelo cargado correctamente en GPU")

# ── Funciones auxiliares ──────────────────────────────────────────────────────
def crear_mascara_region(ancho, alto, region_rel):
    """
    Crea la máscara de ENTRADA (caja delimitadora) para guiar a Stable Diffusion.
    OJO: Esta NO es la máscara final de entrenamiento para la U-Net.
    """
    mascara = Image.new("RGB", (ancho, alto), "black")
    x0 = int(region_rel[0] * ancho)
    y0 = int(region_rel[1] * alto)
    x1 = int(region_rel[2] * ancho)
    y1 = int(region_rel[3] * alto)

    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(mascara)
    draw.rectangle([x0, y0, x1, y1], fill="white")
    return mascara, (x0, y0, x1, y1)

# ── Cargar imágenes ───────────────────────────────────────────────────────────
imagenes = sorted(DIRECTORIO_REAL.glob("*.png"))
print(f"\n[2/4] Imágenes reales encontradas: {len(imagenes)}")

if len(imagenes) == 0:
    print("ERROR: No hay imágenes en data/raw/real/")
    exit(1)

# ── Generar inpaintings y máscaras ground-truth ───────────────────────────────
print(f"[3/4] Generando ediciones locales y extrayendo máscaras...")

exitosas = 0
errores  = 0
registro = []

nombres_regiones = list(REGIONES_FACIALES.keys())
generador = torch.Generator("cuda").manual_seed(SEMILLA)

for i, ruta_real in enumerate(tqdm(imagenes, desc="Inpainting", unit="img")):
    ruta_destino = DIRECTORIO_FAKE / ruta_real.name
    ruta_mascara = DIRECTORIO_MASCARAS / ruta_real.name

    # Si ya existe la imagen falsa Y la máscara, saltar (reanudación segura)
    if ruta_destino.exists() and ruta_mascara.exists():
        exitosas += 1
        registro.append({"imagen": ruta_real.name, "estado": "ya_existia"})
        continue

    try:
        # 1. Preparación para SD (requiere formato PIL y 512x512)
        img_original_pil = Image.open(ruta_real).convert("RGB")
        w_orig, h_orig = img_original_pil.size
        img_512 = img_original_pil.resize((512, 512), Image.Resampling.LANCZOS) # Corrección Pillow 10+
        
        nombre_region = random.choice(nombres_regiones)
        region_rel    = REGIONES_FACIALES[nombre_region]
        prompt        = random.choice(PROMPTS)

        mascara_guia, coords = crear_mascara_region(512, 512, region_rel)

        # 2. Generación Sintética (Inferencia)
        resultado_512 = pipeline(
            prompt              = prompt,
            image               = img_512,
            mask_image          = mascara_guia,
            num_inference_steps = PASOS_SD,
            guidance_scale      = GUIA,
            generator           = generador,
        ).images[0]

        # 3. Restaurar resolución original
        resultado_final_pil = resultado_512.resize((w_orig, h_orig), Image.Resampling.LANCZOS)
        
        # ── EXTRACCIÓN FORENSE DE LA MÁSCARA (NUEVO) ──────────────────────────
        # 0. Conversión de formatos
        # Convertir imágenes de formato PIL (RGB) a OpenCV (NumPy BGR) para poder 
        # operar algebraicamente con las matrices de píxeles.
        img_real_cv = cv2.cvtColor(np.array(img_original_pil), cv2.COLOR_RGB2BGR)
        img_fake_cv = cv2.cvtColor(np.array(resultado_final_pil), cv2.COLOR_RGB2BGR)
        
        # 1. Calcular la diferencia absoluta píxel a píxel
        # Dado que el VAE de Stable Diffusion reconstruye toda la imagen (no solo el 
        # cuadro blanco de la guía) y aplica difuminados en los bordes, el rectángulo 
        # original no sirve como etiqueta. Se calcula la diferencia para obtener la 
        # huella forense exacta de la alteración matemática real.
        diferencia = cv2.absdiff(img_real_cv, img_fake_cv)
        # 2. Convertir a escala de grises
        # Reducción de dimensionalidad: colapsamos los 3 canales BGR de la diferencia 
        # en una única matriz de intensidades de 0 a 255.
        diff_gris  = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
        
        # 3. Binarizar por Umbral
        # Se aplica un umbral estricto (> 5 sobre 255). Cualquier pixel modificado 
        # pasa a ser blanco absoluto (255) y el resto negro (0). Esto es vital para 
        # ignorar el ruido imperceptible de decodificación del VAE o las pequeñas 
        # variaciones introducidas por el escalado bidireccional (256 <-> 512).
        _, mascara_binaria = cv2.threshold(diff_gris, 5, 255, cv2.THRESH_BINARY)
        
        # 4. Operaciones morfológicas para agrupar los píxeles editados
        # Apertura (OPEN): Erosión seguida de dilatación para borrar puntos blancos 
        # aislados (ruido estático fuera de la región facial).
        # Cierre (CLOSE): Dilatación seguida de erosión para rellenar huecos negros 
        # dentro de la máscara principal causados por coincidencias casuales de color.
        kernel = np.ones((5,5), np.uint8)
        mascara_limpia = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel)
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)
        # ──────────────────────────────────────────────────────────────────────
        
        # 4. Guardar disco
        resultado_final_pil.save(ruta_destino, "PNG")
        cv2.imwrite(str(ruta_mascara), mascara_limpia)

        exitosas += 1
        registro.append({
            "imagen":  ruta_real.name,
            "estado":  "exitoso",
            "region":  nombre_region,
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
    "detalle":      registro
}

with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
    json.dump(datos_log, f, indent=2, ensure_ascii=False)

print(f"\n[4/4] Resumen:")
print(f"  Inpaintings exitosos:      {exitosas}")
print(f"  Errores:                   {errores}")
print(f"  Total en fake_inpainting/: {len(list(DIRECTORIO_FAKE.glob('*.png')))}")
print(f"  Máscaras extraídas:        {len(list(DIRECTORIO_MASCARAS.glob('*.png')))}")
print(f"  Registro guardado en:      {ARCHIVO_LOG}")
print(f"\nSIGUIENTE PASO:")
print(f"  uv run python scripts/04_generar_sintesis.py")
print(f"  uv run python scripts/04_generar_sintesis_v2.py")