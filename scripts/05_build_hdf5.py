# scripts/05_build_hdf5.py
"""
Empaqueta el dataset final en un único archivo HDF5 con partición
train/val/test (§8.8.3: 70/15/15 para suficiente variación en cada fase).
Incluye imágenes reales (máscara todo-cero) y falsas (máscara generada).
Justificación del formato: análisis forense offline con GPU dedicada (§8.11.1).
"""
import h5py
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

REAL_DIR   = Path("data/raw/real")
FAKE_SWAP  = Path("data/raw/fake_swap")
FAKE_INP   = Path("data/raw/fake_inpaint")
MASK_DIR   = Path("data/processed/masks")
OUT_HDF5   = Path("data/processed/dataset.h5")

TARGET_SIZE = (256, 256)  # Divisible por 16 — requerimiento U-Net (§8.6.2)

def load_img(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    return img.astype(np.uint8)

def load_mask(path):
    if path is None or not path.exists():
        return np.zeros(TARGET_SIZE, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, TARGET_SIZE)
    return (mask > 127).astype(np.uint8)  # Binario estricto (§8.5.4)

# Construir lista de pares (imagen, máscara, label)
samples = []
for fake_dir in [FAKE_SWAP, FAKE_INP]:
    for fake_path in fake_dir.glob("*.png"):
        mask_path = MASK_DIR / fake_path.name
        samples.append((fake_path, mask_path, 1))  # 1 = manipulada

# Muestras reales con máscara todo-cero — balance de clases (§8.8.2)
real_paths = list(REAL_DIR.glob("*.png"))[:len(samples)]
for real_path in real_paths:
    samples.append((real_path, None, 0))  # 0 = auténtica

# Partición estratificada (§8.8.3)
labels = [s[2] for s in samples]
train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=42)
labels_tv = [s[2] for s in train_val]
train, val = train_test_split(train_val, test_size=0.176, stratify=labels_tv, random_state=42)
# 0.176 de 0.85 ≈ 15% del total → partición 70/15/15

splits = {"train": train, "val": val, "test": test}

with h5py.File(OUT_HDF5, "w") as f:
    for split_name, split_data in splits.items():
        n = len(split_data)
        grp = f.create_group(split_name)
        imgs  = grp.create_dataset("images", shape=(n, 256, 256, 3), dtype=np.uint8,
                                   compression="gzip", compression_opts=4)
        masks = grp.create_dataset("masks",  shape=(n, 256, 256),    dtype=np.uint8,
                                   compression="gzip", compression_opts=4)
        lbls  = grp.create_dataset("labels", shape=(n,),             dtype=np.uint8)

        for i, (img_path, mask_path, label) in enumerate(tqdm(split_data, desc=split_name)):
            imgs[i]  = load_img(img_path)
            masks[i] = load_mask(mask_path)
            lbls[i]  = label

    # Metadatos de auditoría forense (§8.11.2 — trazabilidad)
    f.attrs["total_samples"]   = len(samples)
    f.attrs["train_samples"]   = len(train)
    f.attrs["val_samples"]     = len(val)
    f.attrs["test_samples"]    = len(test)
    f.attrs["image_size"]      = "256x256"
    f.attrs["mask_convention"] = "0=authentic, 1=manipulated"

print(f"HDF5 generado: {OUT_HDF5} | Total: {len(samples)} muestras")