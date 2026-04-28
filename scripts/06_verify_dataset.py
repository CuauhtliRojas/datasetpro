# scripts/06_verify_dataset.py
"""
Verifica: distribución de clases, ratio de píxeles positivos por máscara,
e imágenes con máscara vacía (potencial fallo del detector facial).
Justificación: §8.9.1 — diagnosticar desbalance de clases antes de entrenar.
"""
import h5py
import numpy as np
from pathlib import Path

HDF5 = Path("data/processed/dataset.h5")

with h5py.File(HDF5, "r") as f:
    for split in ["train", "val", "test"]:
        labels = f[split]["labels"][:]
        masks  = f[split]["masks"][:]
        n_fake = (labels == 1).sum()
        n_real = (labels == 0).sum()
        
        # Ratio de píxeles positivos (§8.9.1 — desbalance de clases)
        fake_masks = masks[labels == 1]
        pixel_ratios = fake_masks.mean(axis=(1,2))
        empty_masks  = (pixel_ratios == 0).sum()

        print(f"\n{split.upper()}: {len(labels)} muestras")
        print(f"  Reales: {n_real} | Falsas: {n_fake}")
        print(f"  Ratio px positivos (media): {pixel_ratios.mean():.4f}")
        print(f"  Máscaras vacías en fakes: {empty_masks} ({empty_masks/n_fake*100:.1f}%)")