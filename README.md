# Datasetpro — Contruccion de dataset

**Proyecto:** _Detección y localización de manipulaciones
faciales deepfakes mediante segmentaci´on
semántica basada en redes neuronales
convolucionales_
**Intitucion:** SEPI ESIME Culhuacán, IPN —
**Autores:** Castillo Delgado Ángel Ivan, Rojas Lozada Cuauhtli Emiliano
**Asesores:** Dr. Manuel Cedillo Hern´andez Dr. Rodrigo Eduardo Ar´evalo Ancona
**Año:** 2026

---

## Estado actual del pipeline

| # | Script | Estado | Resultado |
|---|--------|--------|-----------|
| 01 | `01_download_ffhq.py` | ✅ Completado | 1,000 PNG en `data/raw/real/` |
| 02 | `02_generate_swaps.py` | ✅ Completado | 109 fakes en `data/raw/fake_swap/` |
| 04 | `04_generate_masks.py` | ⏳ Siguiente paso | Máscaras binarias en `Train_D/` |
| 05 | `05_assemble_dataset.py` | ⏳ Pendiente | Dataset final ensamblado |

> **Nota sobre el paso 02:** De 1,000 imágenes reales, 109 generaron swaps exitosos.
> Las 891 restantes fueron descartadas porque InsightFace no detectó rostro con
> suficiente confianza en imágenes de 128×128px (thumbnails de FFHQ).
> Esto es comportamiento correcto — las imágenes descartadas no contaminan el dataset.
> Para el dataset de producción (50,000 imágenes) se usará FFHQ en resolución completa (1024px).

---

## Requisitos de hardware

| Componente | Mínimo requerido | Probado en |
|------------|-----------------|-----------|
| GPU NVIDIA | RTX 3060 / 6GB VRAM | RTX 4050 Laptop 6GB ✅ |
| RAM | 16 GB | 16 GB DDR5 ✅ |
| Almacenamiento | 20 GB libres | SSD 512 GB ✅ |
| CUDA | 12.1 o superior | 12.5 (driver 556.19) ✅ |
| Python | 3.11 o superior | 3.12 ✅ |
| OS | Windows 10/11 | Windows 11 ✅ |

---

## Instalación desde cero (laptop virgen con GPU NVIDIA)

### 1. Prerequisitos del sistema operativo

**Instalar Python 3.12:**
Descarga desde `https://www.python.org/downloads/` e instala.
Durante la instalación activa la opción **"Add Python to PATH"**.

Verifica:
```powershell
python --version
# Debe imprimir: Python 3.12.x
```

**Instalar uv:**
```powershell
pip install uv
uv --version
```

**Verificar que tu GPU es reconocida:**
```powershell
nvidia-smi
```
Debe mostrar tu GPU y una versión de CUDA >= 12.1.
Si `nvidia-smi` no funciona, instala los drivers NVIDIA desde:
`https://www.nvidia.com/Download/index.aspx`

### 2. Clonar el repositorio

```powershell
git clone https://github.com/TU_USUARIO/datasetpro.git
cd datasetpro
```

### 3. Crear entorno virtual e instalar dependencias

```powershell
uv venv
uv sync
```

> **IMPORTANTE — PyTorch con CUDA:**
> El `pyproject.toml` está configurado para instalar `torch+cu124` automáticamente.
> Si `uv sync` instala la versión CPU por alguna razón, ejecuta manualmente:
> ```powershell
> uv run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```

**Verificar que CUDA funciona:**
```powershell
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```
Debe imprimir `CUDA: True` y el nombre de tu GPU.

### 4. Configurar Kaggle API

1. Crea una cuenta en `https://www.kaggle.com`
2. Ve a `https://www.kaggle.com/settings` → sección API → **Create New Token**
3. Se descarga `kaggle.json` con tu usuario y key
4. Crea la carpeta y coloca el archivo:

```powershell
mkdir C:\Users\TU_USUARIO\.kaggle
# Copia kaggle.json a C:\Users\TU_USUARIO\.kaggle\kaggle.json
```

El archivo debe tener este formato:
```json
{"username":"tu_usuario_kaggle","key":"tu_api_key"}
```

Verifica:
```powershell
Test-Path C:\Users\TU_USUARIO\.kaggle\kaggle.json
# Debe imprimir: True
```

### 5. Descargar inswapper_128.onnx manualmente

InsightFace no puede descargar este archivo automáticamente desde GitHub.
Descárgalo desde HuggingFace (~500 MB):

```
https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
```

Colócalo exactamente aquí:
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
# Debe imprimir: True
```

### 6. Crear estructura de carpetas de datos

```powershell
mkdir data\raw\real
mkdir data\raw\fake_swap
mkdir data\Train_D\images
mkdir data\Train_D\fake_mask
mkdir data\Train_D\original_mask
```

---

## Ejecutar el pipeline completo

Una vez completada la instalación, ejecuta los scripts en este orden exacto.
Usa siempre `uv run python` — nunca actives el entorno manualmente.

### Paso 01 — Descargar imágenes reales FFHQ

```powershell
uv run python scripts/01_download_ffhq.py
```

- Descarga ~2 GB de FFHQ desde Kaggle
- Selecciona 1,000 imágenes aleatoriamente (semilla fija = reproducible)
- Guarda en `data/raw/real/` como PNG numerados (`00000.png` ... `00999.png`)
- Tiempo estimado: 2–5 minutos según conexión

**Verificar:**
```powershell
uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/real').glob('*.png'))), 'imágenes')"
# Debe imprimir: 1000 imágenes
```

### Paso 02 — Generar fakes con face swap

```powershell
uv run python scripts/02_generate_swaps.py
```

- Por cada imagen real detecta el rostro e intercambia identidad con otra imagen
- Imágenes sin rostro detectable son descartadas sin contaminar el dataset
- Primera ejecución descarga `buffalo_l` (~280 MB) automáticamente
- Guarda fakes en `data/raw/fake_swap/` y log en `data/raw/swap_log.json`
- Tiempo estimado: 1–3 minutos para 1,000 imágenes con GPU

> **Por qué no todos los swaps son exitosos:**
> InsightFace requiere que el rostro sea claramente visible y de tamaño suficiente.
> Las imágenes thumbnail (128×128px) frecuentemente no pasan este umbral.
> Para el dataset de producción se usará FFHQ en resolución completa (256px o 1024px).

**Verificar:**
```powershell
uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/fake_swap').glob('*.png'))), 'fakes generados')"
```

### Paso 04 — Generar máscaras binarias

```powershell
uv run python scripts/04_generate_masks.py
```

- Resta imagen real − imagen fake píxel a píxel (absdiff)
- Aplica umbralización + dilatación morfológica (acotamiento próximo §8.8.3)
- Genera dos máscaras por par: `fake_mask` y `original_mask`
- Solo procesa los pares que existen en ambas carpetas (real + fake)
- Tiempo estimado: menos de 1 minuto para 109 pares

**Verificar:**
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
uv run python scripts/05_assemble_dataset.py
```

- Copia las imágenes fake a `Train_D/images/`
- Verifica que las tres carpetas tienen exactamente los mismos archivos
- Imprime resumen de consistencia del dataset

### Verificación final

```powershell
uv run python scripts/06_verify_dataset.py
```

Debe mostrar las 5 carpetas con el mismo número de archivos y marcar el siguiente paso.

---

## Estructura del repositorio

```
datasetpro/
├── data/                        # NO está en git (.gitignore)
│   ├── raw/
│   │   ├── real/                ← imágenes reales FFHQ
│   │   ├── fake_swap/           ← fakes generados por inswapper
│   │   └── swap_log.json        ← log detallado del paso 02
│   └── Train_D/                 ← dataset final para entrenamiento
│       ├── images/              ← imagen fake (entrada al modelo)
│       ├── fake_mask/           ← máscara región manipulada (blanco=manipulado)
│       └── original_mask/       ← máscara región auténtica (blanco=auténtico)
├── scripts/                     ← pipeline de construcción de datos
│   ├── 01_download_ffhq.py
│   ├── 02_generate_swaps.py
│   ├── 04_generate_masks.py
│   ├── 05_assemble_dataset.py
│   └── 06_verify_dataset.py
├── src/                         ← arquitectura del modelo
│   ├── model.py                 ← DualSegmentationModel (encoder + 2 decoders)
│   ├── train.py                 ← bucle de entrenamiento
│   ├── test.py                  ← evaluación y generación de overlays
│   ├── dataset.py               ← SegmentationDataset
│   ├── losses.py                ← dice_loss, iou_loss, BCE compuesta
│   └── metrics.py               ← IoU, Dice, Recall por píxel
├── notebooks/
│   └── eda_dataset.ipynb        ← análisis exploratorio de distribución de máscaras
├── pyproject.toml               ← dependencias con índice pytorch-cuda configurado
├── uv.lock                      ← versiones exactas para reproducibilidad total
└── README.md
```

---

## Qué produce el pipeline

```
imagen_real.png ──► InsightFace/inswapper ──► imagen_fake.png
                                                     │
imagen_real.png ─────────────────────────────────────┤
        │                                            │
        └──► absdiff + umbral + dilatación ──► fake_mask.png      (zona swap = blanco)
                                               original_mask.png   (zona auténtica = blanco)
```

El modelo `DualSegmentationModel` recibe `imagen_fake` y predice simultáneamente
`fake_mask` y `original_mask` mediante dos decoders independientes sobre un encoder compartido.

---

## Convención de máscaras

| Carpeta | Valor 255 (blanco) | Valor 0 (negro) |
|---------|-------------------|-----------------|
| `fake_mask/` | Región manipulada por el swap | Todo lo demás |
| `original_mask/` | Región auténtica del rostro | Todo lo demás |

Ambas máscaras son PNG en escala de grises del mismo tamaño que la imagen de entrada (128×128px).

---

## Justificación de decisiones técnicas

| Decisión | Justificación en el documento |
|----------|-------------------------------|
| **FFHQ como fuente de reales** | Diversidad demográfica, alta resolución, sin sesgo de actor (§8.8.1) |
| **inswapper_128 para fakes** | Cubre reemplazo de identidad (§8.1.1), introduce artefactos de borde y fotométricos detectables por U-Net (§8.7.1), ejecutable offline sin API externa (§8.11.1) |
| **Descartar imágenes sin rostro** | Evita contaminar el dataset con pares de diferencia cero que generarían máscaras vacías (§8.9.1) |
| **Dilatación morfológica en máscaras** | Produce acotamiento próximo con tolerancia de borde declarada en §8.8.3 y §5 |
| **Carpetas directas vs HDF5** | `train_.py` lee carpetas directamente; HDF5 añade complejidad innecesaria para <50k imágenes |
| **Semilla fija SEED=42** | Reproducibilidad del emparejamiento real↔fuente entre dispositivos (§8.8.3) |
| **BCE + Dice como función de pérdida** | Responde al desbalance de clases: región manipulada es minoría (§8.9.2, §8.9.3) |

---

## Escalar a 50,000 imágenes (dataset de producción)

Para el dataset final descrito en §4.2 del documento:

1. Descargar FFHQ en **256px o 1024px** (no thumbnails de 128px)
   para que InsightFace detecte rostros en >90% de las imágenes
2. Cambiar `TARGET_COUNT = 25000` en `01_download_ffhq.py`
3. Añadir `03_generate_inpainting.py` para cubrir tipología de edición local (§8.1.1)
4. Resultado esperado: 25,000 reales + 25,000 fakes balanceados

---

## Troubleshooting

| Error | Causa | Solución |
|-------|-------|----------|
| `CUDA: False` tras instalar torch | uv instaló versión CPU | `uv run python -m pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `Failed downloading inswapper_128.onnx` | GitHub bloquea descargas automáticas | Descargar manualmente desde HuggingFace (ver Paso 5) |
| `891 sin rostro detectable` | Thumbnails 128px muy pequeños para el detector | Esperado en prueba de pipeline; usar resolución mayor en producción |
| `pip.exe not found` en .venv | uv no genera pip.exe por defecto | Usar siempre `uv run python` en lugar de activar el entorno |
| `where python` no devuelve nada | PATH no configurado en esta sesión | Usar `uv run python` — no requiere PATH configurado |
| `AssertionError: Torch not compiled with CUDA` | torch+cpu instalado en lugar de torch+cu124 | Ver primer troubleshooting arriba |
