# DeepShield — Dataset Construction Pipeline

**Paper:** _DeepShield: A Dual-Decoder Semantic Segmentation Software for Deepfake Face Detection and Digital Identity Protection_  
**Institution:** SEPI ESIME Culhuacán, IPN — Facultad de Ingeniería, UNAM  
**Authors:** Rodrigo Eduardo Arévalo-Ancona, Manuel Cedillo-Hernández, Francisco García-Ugalde  
**Year:** 2026

---

## ¿Qué logramos con el Step 01?

Al ejecutar `scripts/01_download_ffhq.py` descargamos **1,000 imágenes reales** del dataset FFHQ (Flickr-Faces-HQ) desde Kaggle. Estas imágenes son la **base auténtica** del pipeline: rostros reales de alta calidad que después serán manipulados para crear los pares (imagen_fake, máscara_fake, máscara_original) que necesita el modelo.

FFHQ fue elegido porque cumple los requisitos del marco teórico (§8.8.1): diversidad demográfica real, alta resolución original (1024px), sin sesgos de identidad de actor.

---

## Estado actual del proyecto

| Script                   | Estado                                   | Resultado                     |
| ------------------------ | ---------------------------------------- | ----------------------------- |
| `01_download_ffhq.py`    | ✅ Completado                            | 1,000 PNG en `data/raw/real/` |
| `02_generate_swaps.py`   | ⏳ Pendiente — requiere GPU (Laptop)     | Fakes por reemplazo de rostro |
| `04_generate_masks.py`   | ⏳ Pendiente — depende del paso anterior | Máscaras binarias automáticas |
| `05_assemble_dataset.py` | ⏳ Pendiente — paso final                | Carpetas `Train_D/` listas    |

---

## Estructura del repositorio

```
datasetpro/
├── data/
│   ├── raw/
│   │   ├── real/          ← 1,000 imágenes reales FFHQ (COMPLETADO)
│   │   └── fake_swap/     ← fakes generados por SimSwap (PENDIENTE)
│   └── Train_D/           ← dataset final para entrenamiento
│       ├── images/        ← imagen fake (entrada al modelo)
│       ├── original_mask/ ← máscara región auténtica (píxeles reales = blanco)
│       └── fake_mask/     ← máscara región manipulada (píxeles falsos = blanco)
├── scripts/               ← pipeline de construcción de datos
├── src/                   ← arquitectura del modelo (DualSegmentationModel)
├── notebooks/             ← análisis exploratorio
├── pyproject.toml
└── uv.lock
```

---

## Requisitos de hardware

| Tarea                          | Dispositivo recomendado             |
| ------------------------------ | ----------------------------------- |
| Descarga FFHQ (`01_`)          | PC escritorio (sin GPU) ✅          |
| Generación de fakes (`02_`)    | **Laptop RTX 4050** (requiere CUDA) |
| Generación de máscaras (`04_`) | Laptop RTX 4050                     |
| Ensamblado final (`05_`)       | Cualquiera                          |
| Entrenamiento (`train_.py`)    | **Laptop RTX 4050**                 |

---

## Cómo continuar desde la Laptop

### 1. Clonar / sincronizar el repositorio

```powershell
# Si usas Git (recomendado)
git clone https://github.com/TU_USUARIO/datasetpro.git
cd datasetpro

# O copia manual la carpeta completa por USB/red
# IMPORTANTE: la carpeta data/ NO está en git (.gitignore)
# Copia data/raw/real/ manualmente a la laptop
```

### 2. Configurar el entorno en la Laptop

```powershell
# Dentro de la carpeta datasetpro
uv venv
.venv\Scripts\activate
uv sync
```

### 3. Verificar CUDA en la Laptop

```powershell
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Debe imprimir:

```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

### 4. Copiar las imágenes reales

La carpeta `data/raw/real/` con las 1,000 imágenes **no está en git**. Cópiala manualmente desde el PC de escritorio a la laptop antes de continuar. Puedes usar USB, red local, o Google Drive.

### 5. Ejecutar el pipeline completo (en orden)

```powershell
# Paso 2: Generar fakes con SimSwap (requiere GPU)
python scripts/02_generate_swaps.py

# Paso 4: Generar máscaras binarias automáticas
python scripts/04_generate_masks.py

# Paso 5: Ensamblar Train_D/ final
python scripts/05_assemble_dataset.py

# Verificación opcional
python scripts/06_verify_dataset.py
```

---

## ¿Qué produce el pipeline?

Cada imagen real pasa por este flujo:

```
imagen_real.png  ──►  SimSwap  ──►  imagen_fake.png
                                         │
imagen_real.png ─────────────────────────┤
         │                               │
         └──► sustracción + morfología ──►  fake_mask.png   (zona manipulada = blanco)
                                             original_mask.png (zona auténtica = blanco)
```

El modelo `DualSegmentationModel` recibe `imagen_fake` y predice simultáneamente `fake_mask` y `original_mask`.

---

## Convención de máscaras

| Valor de píxel | Significado                                                                |
| -------------- | -------------------------------------------------------------------------- |
| `255` (blanco) | Región manipulada (en `fake_mask`) / Región auténtica (en `original_mask`) |
| `0` (negro)    | Fondo / región no relevante                                                |

Las máscaras son imágenes PNG en escala de grises, mismo tamaño que la imagen de entrada.

---

## Dependencias clave

| Librería                          | Uso                                                |
| --------------------------------- | -------------------------------------------------- |
| `insightface` + `onnxruntime-gpu` | Generación de face swaps (SimSwap)                 |
| `opencv-python-headless`          | Sustracción de imágenes y morfología para máscaras |
| `torch` + `torchvision`           | Modelo y entrenamiento                             |
| `Pillow`                          | Carga y guardado de imágenes                       |
| `tqdm`                            | Barras de progreso                                 |

---

## Notas importantes

- El dataset de 1,000 imágenes es una **prueba de pipeline**. Para producción se escala a 50,000 (§4.2 del documento de tesis).
- La carpeta `data/` está en `.gitignore` — nunca se sube a GitHub. Transfiere los datos entre dispositivos manualmente.
- El archivo `uv.lock` garantiza que ambos dispositivos usen exactamente las mismas versiones de librerías.
- SimSwap requiere descargar el modelo `inswapper_128.onnx` (~500 MB) la primera vez que se ejecuta. Se descarga automáticamente en la primera ejecución de `02_generate_swaps.py`.
