# Datasetpro — Construcción de Dataset

**Proyecto:** _Detección y localización de manipulaciones faciales deepfakes mediante
segmentación semántica basada en redes neuronales convolucionales_
**Institución:** SEPI ESIME Culhuacán, Instituto Politécnico Nacional
**Autores:** Castillo Delgado Ángel Ivan, Rojas Lozada Cuauhtli Emiliano
**Asesores:** Dr. Manuel Cedillo Hernández, Dr. Rodrigo Eduardo Arévalo Ancona
**Año:** 2026

---

## ¿Qué es este repositorio?

Pipeline automatizado para construir el dataset de entrenamiento del modelo
`DualSegmentationModel` — una red neuronal de doble decoder que detecta y
localiza manipulaciones faciales deepfake mediante segmentación semántica binaria.

El pipeline cubre las **4 tipologías de deepfake** definidas en §8.1.1 del documento:

| Tipología | Script | Descripción |
|-----------|--------|-------------|
| Reemplazo de rostro | `02_generar_swaps.py` | Sustituye identidad completa, fondo real |
| Edición local | `03_generar_inpainting.py` | Modifica zona acotada (boca, ojos, nariz) |
| Síntesis completa | `07_generar_sintesis.py` | Rostro 100% sintético sin persona real |
| Recreación | `08_generar_reenactment.py` | Deforma geometría facial manteniendo identidad |

---

## Estado actual del pipeline (prueba con 1,000 imágenes)

| # | Script | Estado | Resultado |
|---|--------|--------|-----------|
| 01 | `01_descargar_ffhq.py` | ✅ Completado | 1,000 PNG en `data/raw/real/` |
| 02 | `02_generar_swaps.py` | ✅ Completado | 109 fakes en `data/raw/fake_swap/` |
| 03 | `03_generar_inpainting.py` | ✅ Completado | 1,000 fakes en `data/raw/fake_inpainting/` |
| 04 | `04_generar_mascaras.py` | ✅ Completado | Máscaras en `data/Train_D/` |
| 05 | `05_ensamblar_dataset.py` | ✅ Completado | Dataset ensamblado en `Train_D/` |
| 06 | `06_verificar_dataset.py` | ✅ Completado | Diagnóstico de estado |
| 07 | `07_generar_sintesis.py` | ✅ Completado | 1,000 fakes en `data/raw/fake_sintesis/` |
| 08 | `08_generar_reenactment.py` | ✅ Completado | 1,000 fakes en `data/raw/fake_reenactment/` |

> **Nota sobre el paso 02:** De 1,000 imágenes reales solo 109 generaron swaps exitosos.
> InsightFace requiere rostros claramente visibles — los thumbnails de 128×128px de FFHQ
> no cumplen este umbral en la mayoría de los casos. Las imágenes sin rostro detectable
> se descartan para no contaminar el dataset (§8.9.1). En producción se usa FFHQ 256px+.

---

## Requisitos de hardware

| Componente | Mínimo para producción | Probado en |
|------------|------------------------|-----------|
| GPU NVIDIA | RTX 3060 / 6 GB VRAM | RTX 4050 Laptop 6 GB ✅ |
| RAM | 16 GB | 16 GB DDR5 ✅ |
| Almacenamiento | 50 GB libres | SSD 512 GB ✅ |
| CUDA | 12.1 o superior | 12.5 (driver 556.19) ✅ |
| Python | 3.11 o superior | 3.12 ✅ |
| Sistema operativo | Windows 10/11 | Windows 11 ✅ |

> Sin GPU: solo puedes ejecutar los pasos 01, 04, 05 y 06.
> Los pasos 02, 03, 07 y 08 requieren GPU NVIDIA obligatoriamente.

---

## Instalación desde cero en una laptop nueva con GPU NVIDIA

Sigue estos pasos en orden exacto antes de ejecutar cualquier script.

### 1. Instalar Python 3.12

Descarga el instalador desde `https://www.python.org/downloads/`.
Durante la instalación activa **"Add Python to PATH"** (casilla al inicio del instalador).

Verifica en PowerShell:
```powershell
python --version
# Resultado esperado: Python 3.12.x
```

### 2. Instalar uv (gestor de entornos)

```powershell
pip install uv
uv --version
# Resultado esperado: uv 0.x.x
```

### 3. Verificar que la GPU es reconocida por Windows

```powershell
nvidia-smi
```

Debe mostrar el nombre de tu GPU y `CUDA Version: 12.x` o superior.
Si el comando no existe, instala los drivers NVIDIA desde:
`https://www.nvidia.com/Download/index.aspx`

### 4. Clonar el repositorio

```powershell
git clone https://github.com/TU_USUARIO/datasetpro.git
cd datasetpro
```

### 5. Crear entorno virtual e instalar dependencias

```powershell
uv venv
uv sync
```

`uv sync` lee el `pyproject.toml` e instala todas las dependencias incluyendo
`torch+cu124` automáticamente. Si después de sincronizar CUDA no funciona,
reinstala PyTorch manualmente:

```powershell
uv run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verifica que CUDA funciona:
```powershell
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Resultado esperado:
# CUDA: True
# GPU: NVIDIA GeForce RTX XXXX
```

> **Importante:** usa siempre `uv run python` para ejecutar scripts.
> No actives el entorno con `activate` — puede apuntar al Python del sistema.

### 6. Configurar Kaggle API (necesario para paso 01)

1. Crea una cuenta en `https://www.kaggle.com`
2. Ve a `https://www.kaggle.com/settings` → sección **API** → **Create New Token**
3. Se descarga automáticamente `kaggle.json`
4. Coloca el archivo en:

```powershell
mkdir C:\Users\TU_USUARIO\.kaggle
# Copia kaggle.json a esa carpeta
```

El archivo debe contener:
```json
{"username":"tu_usuario_kaggle","key":"KGAT_xxxxxxxxxxxx"}
```

Verifica:
```powershell
Test-Path C:\Users\TU_USUARIO\.kaggle\kaggle.json
# Resultado esperado: True
```

### 7. Descargar inswapper_128.onnx manualmente (necesario para paso 02)

InsightFace no puede descargarlo automáticamente — GitHub bloquea archivos grandes
en scripts. Descárgalo manualmente desde HuggingFace (~500 MB):

```
https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
```

Colócalo en:
```
C:\Users\TU_USUARIO\.insightface\models\inswapper_128.onnx
```

Crea la carpeta si no existe:
```powershell
mkdir C:\Users\TU_USUARIO\.insightface\models
```

Verifica:
```powershell
Test-Path C:\Users\TU_USUARIO\.insightface\models\inswapper_128.onnx
# Resultado esperado: True
```

### 8. Crear estructura de carpetas de datos

```powershell
mkdir data\raw\real
mkdir data\raw\fake_swap
mkdir data\raw\fake_inpainting
mkdir data\raw\fake_sintesis
mkdir data\raw\fake_reenactment
mkdir data\Train_D\images
mkdir data\Train_D\fake_mask
mkdir data\Train_D\original_mask
```

---

## Ejecutar el pipeline completo

Ejecuta los scripts en este orden exacto. Cada script indica el siguiente paso al terminar.

### Paso 01 — Descargar imágenes reales FFHQ

```powershell
uv run python scripts/01_descargar_ffhq.py
```

Qué hace: descarga el dataset FFHQ desde Kaggle y selecciona 1,000 imágenes
aleatoriamente con semilla fija (reproducible en cualquier dispositivo).

- Descarga: ~2 GB
- Tiempo estimado: 3–5 minutos según conexión
- Salida: `data/raw/real/` con archivos `00000.png` ... `00999.png`

Verificar:
```powershell
uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/real').glob('*.png'))), 'imágenes reales')"
# Resultado esperado: 1000 imágenes reales
```

### Paso 02 — Generar fakes por reemplazo de rostro ⚠️ Requiere GPU

```powershell
uv run python scripts/02_generar_swaps.py
```

Qué hace: por cada imagen real detecta el rostro con InsightFace buffalo_l
e intercambia la identidad con el rostro de otra imagen del dataset.
Imágenes sin rostro detectable se descartan automáticamente.

- Primera ejecución descarga buffalo_l (~280 MB) automáticamente
- inswapper_128.onnx debe estar descargado manualmente (ver paso 7 de instalación)
- Tiempo estimado: 2–5 minutos para 1,000 imágenes con GPU
- Salida: `data/raw/fake_swap/` y `data/raw/swap_log.json`

Verificar:
```powershell
uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/fake_swap').glob('*.png'))), 'swaps generados')"
```

### Paso 03 — Generar fakes por edición local ⚠️ Requiere GPU (~90 min)

```powershell
uv run python scripts/03_generar_inpainting.py
```

Qué hace: edita zonas acotadas del rostro (boca, ojos, nariz) usando
Stable Diffusion Inpainting. La zona editada varía por imagen para diversidad.

- Primera ejecución descarga SD Inpainting (~5 GB) desde HuggingFace
- Tiempo estimado: 5 segundos/imagen × 1,000 = ~90 minutos con RTX 4050
- Salida: `data/raw/fake_inpainting/` y `data/raw/inpainting_log.json`

Verificar:
```powershell
uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/fake_inpainting').glob('*.png'))), 'inpaintings generados')"
```

### Paso 04 — Generar máscaras binarias

```powershell
uv run python scripts/04_generar_mascaras.py
```

Qué hace: por cada par (imagen_real, imagen_fake) genera dos máscaras binarias
por sustracción automática píxel a píxel, umbralización y dilatación morfológica.

- No requiere GPU
- Tiempo estimado: menos de 1 minuto para 109 pares
- Salida: `data/Train_D/fake_mask/` y `data/Train_D/original_mask/`

Verificar:
```powershell
uv run python -c "
from pathlib import Path
fm = len(list(Path('data/Train_D/fake_mask').glob('*.png')))
om = len(list(Path('data/Train_D/original_mask').glob('*.png')))
print(f'fake_mask: {fm} | original_mask: {om}')
"
```

### Paso 05 — Ensamblar Train_D/

```powershell
uv run python scripts/05_ensamblar_dataset.py
```

Qué hace: copia las imágenes fake a `Train_D/images/` y verifica que las tres
carpetas requeridas por `train.py` tienen exactamente los mismos archivos.

- No requiere GPU
- Tiempo estimado: menos de 1 minuto
- Salida: `data/Train_D/images/` — dataset listo para entrenamiento

### Paso 06 — Verificar estado del dataset

```powershell
uv run python scripts/06_verificar_dataset.py
```

Qué hace: muestra el conteo de archivos en cada carpeta del pipeline e indica
el siguiente paso a ejecutar. Funciona en cualquier momento y dispositivo.

### Paso 07 — Generar fakes por síntesis completa ⚠️ Requiere GPU (~90 min)

```powershell
uv run python scripts/07_generar_sintesis.py
```

Qué hace: genera rostros 100% sintéticos sin persona real de origen usando
Stable Diffusion text-to-image. Usa el modelo ya descargado en el paso 03.

- Sin descarga adicional si el paso 03 ya fue ejecutado
- Tiempo estimado: ~90 minutos para 1,000 imágenes con RTX 4050
- Salida: `data/raw/fake_sintesis/` y `data/raw/sintesis_log.json`

### Paso 08 — Generar fakes por recreación facial

```powershell
uv run python scripts/08_generar_reenactment.py
```

Qué hace: aplica transformaciones afines a cada imagen real para simular
cambios de pose y expresión manteniendo la identidad original.

- No requiere descarga adicional
- Tiempo estimado: menos de 2 minutos para 1,000 imágenes
- Salida: `data/raw/fake_reenactment/` y `data/raw/reenactment_log.json`

---

## Qué produce el pipeline

```
imagen_real.png ──► inswapper_128     ──► fake_swap/imagen.png
imagen_real.png ──► SD Inpainting     ──► fake_inpainting/imagen.png
                ──► SD text-to-image  ──► fake_sintesis/imagen.png
imagen_real.png ──► Transformación afín──► fake_reenactment/imagen.png

Por cada par (real, fake):
imagen_real.png ─┐
                 ├──► absdiff + umbral + dilatación ──► Train_D/fake_mask/imagen.png
imagen_fake.png ─┘                                      Train_D/original_mask/imagen.png
```

El modelo `DualSegmentationModel` recibe `Train_D/images/imagen.png` y predice
simultáneamente `fake_mask` y `original_mask` mediante dos decoders independientes.

---

## Estructura del repositorio

```
datasetpro/
├── data/                             # NO está en git (.gitignore)
│   ├── raw/
│   │   ├── real/                     ← 1,000 imágenes reales FFHQ
│   │   ├── fake_swap/                ← fakes por reemplazo de rostro
│   │   ├── fake_inpainting/          ← fakes por edición local
│   │   ├── fake_sintesis/            ← fakes por síntesis completa
│   │   ├── fake_reenactment/         ← fakes por recreación facial
│   │   ├── swap_log.json             ← log del paso 02
│   │   ├── inpainting_log.json       ← log del paso 03
│   │   ├── sintesis_log.json         ← log del paso 07
│   │   └── reenactment_log.json      ← log del paso 08
│   └── Train_D/                      ← dataset final para entrenamiento
│       ├── images/                   ← imagen fake (entrada al modelo)
│       ├── fake_mask/                ← máscara región manipulada (blanco=manipulado)
│       └── original_mask/            ← máscara región auténtica (blanco=auténtico)
├── scripts/                          ← pipeline de construcción de datos
│   ├── 01_descargar_ffhq.py          ← Paso 1: imágenes reales
│   ├── 02_generar_swaps.py           ← Paso 2: reemplazo de rostro
│   ├── 03_generar_inpainting.py      ← Paso 3: edición local
│   ├── 04_generar_mascaras.py        ← Paso 4: máscaras binarias
│   ├── 05_ensamblar_dataset.py       ← Paso 5: Train_D/ final
│   ├── 06_verificar_dataset.py       ← Diagnóstico en cualquier momento
│   ├── 07_generar_sintesis.py        ← Paso 7: síntesis completa
│   └── 08_generar_reenactment.py     ← Paso 8: recreación facial
├── src/                              ← arquitectura del modelo
│   ├── model.py                      ← DualSegmentationModel
│   ├── train.py                      ← bucle de entrenamiento
│   ├── test.py                       ← evaluación y métricas
│   ├── dataset.py                    ← SegmentationDataset
│   ├── losses.py                     ← dice_loss, iou_loss
│   └── metrics.py                    ← IoU, Dice, Recall
├── notebooks/
│   └── eda_dataset.ipynb             ← análisis exploratorio
├── pyproject.toml                    ← dependencias con torch+cu124
├── uv.lock                           ← versiones exactas (reproducibilidad)
└── README.md
```

---

## Convención de máscaras

| Carpeta | Valor 255 — blanco | Valor 0 — negro |
|---------|-------------------|-----------------|
| `fake_mask/` | Región manipulada | Resto de la imagen |
| `original_mask/` | Región auténtica | Resto de la imagen |

Ambas máscaras son PNG en escala de grises, tamaño 128×128px.

---

## Justificación de decisiones técnicas

| Decisión | Sección del documento |
|----------|-----------------------|
| FFHQ como fuente de imágenes reales | §8.8.1 — diversidad demográfica sin sesgo de actor |
| Descartar imágenes sin rostro detectable | §8.9.1 — evitar contaminación con máscaras vacías |
| 4 tipologías de deepfake | §8.1.1 — cobertura de reemplazo, edición, síntesis y recreación |
| Dilatación morfológica en máscaras | §8.8.3 y §5 — acotamiento próximo con tolerancia de borde |
| Carpetas directas en lugar de HDF5 | train.py lee carpetas directamente; HDF5 innecesario para <50k imágenes |
| Semilla fija SEED=42 | §8.8.3 — reproducibilidad entre dispositivos |
| BCE + Dice como función de pérdida | §8.9.2 y §8.9.3 — desbalance región manipulada vs auténtica |
| inswapper_128 para reemplazo | §8.7.1 — introduce bordes de mezcla y halos detectables |
| SD Inpainting para edición local | §8.7.1 — inconsistencias fotométricas en zona editada |
| SD text-to-image para síntesis | §8.1.1 — identidad completamente inventada sin referencia real |

---

## Escalar a 50,000 imágenes (producción)

Para el dataset final descrito en §4.2:

1. Descargar FFHQ en **256px o 1024px** — InsightFace detecta rostros en >90% de los casos
2. Cambiar `TOTAL_IMAGENES = 12500` en `01_descargar_ffhq.py` (×4 tipologías = 50,000)
3. Activar el detector Haar en `04_generar_mascaras.py` (sección comentada en el script)
4. Cambiar `UMBRAL = 15` y `DILATACION = 18` en `04_generar_mascaras.py`
5. Resultado esperado: ~12,500 pares por tipología = 50,000 tripletas totales

---

## Troubleshooting

| Error | Causa probable | Solución |
|-------|---------------|----------|
| `CUDA: False` después de `uv sync` | uv instaló torch+cpu | `uv run python -m pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `Failed downloading inswapper_128.onnx` | GitHub bloquea descargas grandes | Descargar manualmente desde HuggingFace (ver paso 7 instalación) |
| `pip.exe not found` en .venv | uv no genera pip.exe | Usar siempre `uv run python` en lugar de activar el entorno |
| `where python` no devuelve nada | PATH no configurado en la sesión | Usar `uv run python` — no requiere PATH configurado |
| `cv2.error: The function is not implemented` | opencv-headless sin soporte GUI | Usar PIL para visualización — opencv-headless no tiene ventanas |
| `891 sin rostro detectable` en paso 02 | Thumbnails 128px muy pequeños | Esperado — usar resolución mayor en producción |
| `Repository Not Found` en HuggingFace | Repo privado o URL incorrecta | Verificar token HuggingFace y URL del modelo |
| Inpainting muy lento | SD requiere muchos pasos | Reducir `PASOS_SD = 20` en `03_generar_inpainting.py` |