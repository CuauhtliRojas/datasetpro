# Modelos Alternativos por Tipología de Deepfake

**Proyecto:** Detección y localización de manipulaciones faciales deepfakes mediante segmentación semántica basada en redes neuronales convolucionales
**Institución:** SEPI ESIME Culhuacán, IPN
**Autores:** Castillo Delgado Ángel Ivan, Rojas Lozada Cuauhtli Emiliano
**Año:** 2026

---

## Tipología 1 — Reemplazo de rostro

Definición 8.1.1: sustituye el rostro completo de una persona por el de otra. La identidad cambia pero el fondo, ropa y contexto permanecen reales.
Artefactos esperados 8.7.1: discontinuidades en bordes de mezcla, halos en contorno facial, inconsistencias fotométricas entre rostro pegado y escena.

| Modelo | Calidad (1-5) | Costo | Dónde ejecutar | VRAM requerida | Por qué es mejor que el actual | Limitación principal | Automatizable por script | Estado del repo | Relevancia en documento |
|--------|--------------|-------|----------------|---------------|-------------------------------|---------------------|--------------------------|----------------|------------------------|
| inswapper_128 — ACTUAL | 3 | Gratis | Laptop local | 1.5 GB | Modelo base funcional, ya integrado en el pipeline | Artefactos visibles, detecta solo 11% de rostros a 128px | Sí | Activo | 8.1.1, 8.7.1 |
| FaceFusion | 4 | Gratis | Laptop local | 2 GB | Wrapper moderno de inswapper con post-procesamiento automático, mejora calidad de bordes y blending sin cambiar arquitectura base | Instalación separada fuera de uv, requiere Python 3.10 | Sí, via subprocess | Activo 2024 | 8.1.1, 8.7.1 |
| SimSwap | 4 | Gratis | Colab T4 | 3 GB | Mejor blending que inswapper, artefactos de borde más sutiles y realistas, mayor valor forense para el entrenamiento del modelo | Instalación compleja en Windows, dependencias conflictivas con el entorno actual | Sí, en Colab | Activo | 8.1.1, 8.7.1 |
| DeepFaceLab | 5 | Gratis | Laptop local | 4 GB | Mejor calidad de mezcla open-source disponible, artefactos más naturales e imperceptibles para el ojo humano | No automatizable por script Python puro, requiere GUI para configuración inicial | No | Activo | 8.1.1, 8.7.1 |
| Runway Gen-3 Alpha | 5 | 15 USD/mes | API externa | N/A — cloud | Calidad comercial, artefactos imperceptibles para humanos, ideal para casos difíciles del dataset | API de pago, datos salen del dispositivo, no cumple modo offline 8.11.1 | Sí, via API | Activo comercial | 8.1.1, 8.7.1, 8.11.1 |
| Akool FaceSwap API | 5 | 0.01 USD/imagen | API externa | N/A — cloud | Mejor calidad de mezcla disponible actualmente, usado en producción comercial, soporta alta resolución | Costo por imagen escala con el tamaño del dataset, datos salen del dispositivo | Sí, via API | Activo comercial | 8.1.1, 8.7.1 |

---

## Tipología 2 — Edición local

Definición 8.1.1: modifica zonas acotadas del rostro sin sustituir la identidad completa. Las regiones más frecuentes son boca, ojos y nariz.
Artefactos esperados 8.7.1: inconsistencias fotométricas en la zona editada, bordes de fusión entre región generada y región auténtica, diferencias de textura de piel.

| Modelo | Calidad (1-5) | Costo | Dónde ejecutar | VRAM requerida | Por qué es mejor que el actual | Limitación principal | Automatizable por script | Estado del repo | Relevancia en documento |
|--------|--------------|-------|----------------|---------------|-------------------------------|---------------------|--------------------------|----------------|------------------------|
| SD Inpainting v1-5 — ACTUAL | 3 | Gratis | Laptop o Colab | 4.2 GB float16 | Modelo base funcional, ya descargado en caché del pipeline | Inconsistencia con contexto facial, resolución nativa 512px limita calidad a 256px | Sí | Activo | 8.1.1, 8.7.1 |
| Stable Diffusion XL Inpainting | 4 | Gratis | Colab A100 | 8 GB | Resolución nativa 1024px, mejor coherencia facial, artefactos más sutiles y fotorrealistas que SD v1-5 | 8 GB VRAM mínimo, no corre en RTX 4050 sin CPU offloading | Sí | Activo 2024 | 8.1.1, 8.7.1 |
| Flux.1 Fill | 5 | Gratis | Colab A100 | 12 GB | Estado del arte 2024 en inpainting, coherencia contextual superior a SD XL, artefactos de borde imperceptibles para humanos | Requiere A100 para inferencia fluida, modelo de 12 GB | Sí | Activo 2024 | 8.1.1, 8.7.1 |
| LaMa Large Mask | 4 | Gratis | Laptop o Colab | 2 GB | Especializado en inpainting coherente, rellena sin artefactos visibles de borde, muy rápido en inferencia | Demasiado bueno: puede no dejar artefactos detectables para el segmentador, contraproducente para el dataset | Sí | Activo | 8.1.1, 8.7.1 |
| DALL-E 3 Inpainting | 5 | 0.04 USD/imagen | API OpenAI | N/A — cloud | Máxima coherencia contextual disponible, artefactos fotorrealistas de alta calidad | Costo por imagen, datos salen del dispositivo, no cumple modo offline 8.11.1 | Sí, via API | Activo comercial | 8.1.1, 8.7.1, 8.11.1 |
| Adobe Firefly Inpainting | 5 | 55 USD/mes | API externa | N/A — cloud | Mejor integración contextual del mercado, artefactos fotorrealistas, soporte de máscara precisa por región | Suscripción Adobe, costo fijo mensual, datos salen del dispositivo | Sí, via API | Activo comercial | 8.1.1, 8.7.1 |

---

## Tipología 3 — Síntesis completa

Definición 8.1.1: genera un rostro completamente inventado sin persona real de origen. No existe imagen de referencia auténtica.
Artefactos esperados 8.7.2: coherencia global alta con inconsistencias locales de textura en regiones de alta frecuencia, artefactos espectrales característicos de GAN o modelos de difusión.

| Modelo | Calidad (1-5) | Costo | Dónde ejecutar | VRAM requerida | Por qué es mejor que el actual | Limitación principal | Automatizable por script | Estado del repo | Relevancia en documento |
|--------|--------------|-------|----------------|---------------|-------------------------------|---------------------|--------------------------|----------------|------------------------|
| SD v1-5 text-to-image — ACTUAL | 3 | Gratis | Laptop o Colab | 4.2 GB float16 | Modelo base funcional, reutiliza caché del paso 03 del pipeline | Rostros genéricos poco fotorrealistas, baja diversidad de identidades reales | Sí | Activo | 8.1.1, 8.7.2 |
| StyleGAN2-ADA FFHQ | 5 | Gratis | Colab T4 | 3 GB | Estándar de referencia académico en detección de deepfakes, directamente citado en FaceForensics++ mencionado en el documento, artefactos GAN documentados en literatura forense | Pesos de 300 MB requieren descarga manual desde servidores NVIDIA, sin instalación pip directa | Sí, con descarga manual | Mantenido por NVIDIA | 8.1.1, 8.7.2, 8.8.1 |
| StyleGAN3 | 5 | Gratis | Colab A100 | 6 GB | Elimina aliasing de StyleGAN2, entrenado en FFHQ, artefactos de alta frecuencia más limpios y relevantes para 8.7.2 | Requiere A100 para inferencia rápida, pesos descarga manual desde NVIDIA | Sí, con descarga manual | Mantenido por NVIDIA | 8.1.1, 8.7.2 |
| Flux.1 Schnell | 5 | Gratis | Colab A100 | 12 GB | Estado del arte 2024 en text-to-image, fotorrealismo superior a SD en rostros, mayor diversidad de identidades sintéticas | Requiere A100, modelo de 12 GB de descarga | Sí | Activo 2024 | 8.1.1, 8.7.2 |
| This Person Does Not Exist — scraping | 5 | Gratis | Web | N/A | StyleGAN2 en producción, exactamente el tipo de síntesis descrito en 8.7.2, imágenes de alta calidad a 1024px disponibles públicamente | Requiere scraping, términos de uso ambiguos, velocidad limitada por rate limiting del sitio | Sí, via selenium | Activo | 8.1.1, 8.7.2 |
| Midjourney v6 | 5 | 10 USD/mes | API Discord | N/A — cloud | Mejor calidad visual disponible para rostros sintéticos, altísima diversidad de identidades, fotorrealismo comercial | No automatizable directamente, requiere interacción via Discord, datos salen del dispositivo | Parcial, via bot | Activo comercial | 8.1.1, 8.7.2 |

---

## Tipología 4 — Recreación facial

Definición 8.1.1: transfiere expresiones o movimientos faciales de una persona a otra manteniendo la identidad original. El rostro es real pero la expresión es sintética.
Artefactos esperados 8.7.1: discontinuidades en expresiones, deformaciones en contorno labial y región periocular, inconsistencias de movimiento en zonas de alto gradiente.

| Modelo | Calidad (1-5) | Costo | Dónde ejecutar | VRAM requerida | Por qué es mejor que el actual | Limitación principal | Automatizable por script | Estado del repo | Relevancia en documento |
|--------|--------------|-------|----------------|---------------|-------------------------------|---------------------|--------------------------|----------------|------------------------|
| Transformación afín OpenCV — ACTUAL | 2 | Gratis | Laptop local | 0.1 GB | Funcional como aproximación geométrica, sin dependencias adicionales | Artefactos demasiado sutiles y no fotorrealistas, valor forense limitado para entrenar el modelo | Sí | N/A — librería estándar | 8.1.1, 8.7.1 |
| LivePortrait | 5 | Gratis | Colab T4 o Laptop | 6 GB | Estado del arte 2024 open-source, mejor calidad que FOMM, repo activo y mantenido, introduce deformaciones faciales realistas en zonas de alto movimiento facial | 6 GB VRAM en local al límite del RTX 4050, más estable en Colab T4 | Sí | Activo 2024 | 8.1.1, 8.7.1 |
| SadTalker | 4 | Gratis | Colab T4 o Laptop | 4 GB | Anima rostros estáticos con landmarks o audio, fácil de instalar via pip, bien documentado, produce artefactos en contorno labial y región periocular exactamente como describe 8.7.1 | Orientado a video, requiere extraer frame representativo para dataset de imagen fija | Sí | Activo | 8.1.1, 8.7.1 |
| First Order Motion Model FOMM | 4 | Gratis | Colab T4 | 3 GB | Referencia académica para reenactment, usado en la construcción de FaceForensics++, artefactos de deformación documentados en literatura | Repo sin mantenimiento activo, checkpoint oficial no disponible en HuggingFace público, difícil instalación | Sí, en Colab | Sin mantenimiento | 8.1.1, 8.7.1 |
| DiffusedHeads | 5 | Gratis | Colab A100 | 16 GB | Estado del arte 2024 en reenactment con modelos de difusión, artefactos más realistas e imperceptibles que FOMM | Requiere A100, implementación compleja, pocos ejemplos de uso programático disponibles | Parcial | Activo 2024 | 8.1.1, 8.7.1 |
| HeyGen Avatar API | 5 | 29 USD/mes | API externa | N/A — cloud | Calidad comercial de video reenactment, alta fidelidad en transferencia de expresión, usado en deepfakes virales reales documentados | Costo mensual fijo, orientado a video no imagen fija, datos salen del dispositivo | Sí, via API | Activo comercial | 8.1.1, 8.7.1 |

---

## Resumen comparativo por escenario de uso

| Escenario | Tipología | Modelo recomendado | Justificación |
|-----------|-----------|-------------------|---------------|
| Prueba de pipeline (500 imgs, local) | Reemplazo | inswapper_128 actual | Ya funciona, sin configuración adicional |
| Prueba de pipeline (500 imgs, local) | Edición local | SD Inpainting v1-5 actual | Ya descargado en caché |
| Prueba de pipeline (500 imgs, local) | Síntesis | StyleGAN2-ADA FFHQ | Estándar académico, corre en Colab T4 gratuito |
| Prueba de pipeline (500 imgs, local) | Recreación | SadTalker | Fácil instalación, produce artefactos relevantes |
| Producción (50,000 imgs, Colab) | Reemplazo | SimSwap | Mejor calidad, automatizable en Colab |
| Producción (50,000 imgs, Colab) | Edición local | SD XL Inpainting | Resolución 1024px, coherencia superior |
| Producción (50,000 imgs, Colab) | Síntesis | StyleGAN2-ADA FFHQ | Referencia académica directamente citada en el documento |
| Producción (50,000 imgs, Colab) | Recreación | LivePortrait | Estado del arte 2024, repo activo |

---

## Estrategia de ejecución recomendada

```
Laptop local (RTX 4050)       Colab T4 gratuito              Colab Pro A100
──────────────────────        ──────────────────────         ──────────────────
01_descargar_ffhq             02_generar_swaps (SimSwap)     03_generar_inpainting (SD XL)
06_generar_mascaras           04_generar_sintesis (SG2-ADA)
07_ensamblar_dataset          05_generar_reenactment (LivePortrait)
08_verificar_dataset
```