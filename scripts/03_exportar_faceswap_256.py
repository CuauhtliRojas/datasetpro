# scripts/03_exportar_faceswap_256.py
# =============================================================================
# Exporta dataset faceswap generado en resolución maestra a 256x256.
#
# Reglas:
#   - Imagen RGB/fake: LANCZOS.
#   - Máscara binaria: NEAREST.
#   - Overlay QA: LANCZOS.
# =============================================================================

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta faceswap 1024 a entrenamiento 256.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/content/drive/MyDrive/datasetpro_colab"),
        help="Raíz del workspace en Google Drive.",
    )
    parser.add_argument(
        "--source_resolution",
        type=int,
        default=1024,
        help="Resolución maestra esperada.",
    )
    parser.add_argument(
        "--target_resolution",
        type=int,
        default=256,
        help="Resolución final de entrenamiento.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir exportaciones existentes.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> object | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_mask(path: Path) -> object | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def write_log(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    workspace = args.workspace
    data_dir = workspace / "data"

    fake_dir = data_dir / "generated" / "fake_swap_1024"
    mask_dir = data_dir / "generated" / "masks_swap_1024"
    overlay_dir = data_dir / "generated" / "overlays_swap_1024"

    export_images = data_dir / "export_256" / "images"
    export_masks = data_dir / "export_256" / "masks"
    export_overlays = data_dir / "export_256" / "overlays"
    log_path = data_dir / "logs" / "export_faceswap_256_log.json"

    for folder in [export_images, export_masks, export_overlays, log_path.parent]:
        ensure_dir(folder)

    fake_images = iter_images(fake_dir)
    if not fake_images:
        print(f"ERROR: No hay imágenes fake en {fake_dir}")
        return 1

    print("=" * 72)
    print("DATASETPRO — EXPORT FACESWAP 256")
    print("=" * 72)
    print(f"Workspace:       {workspace}")
    print(f"Fake source:     {fake_dir}")
    print(f"Mask source:     {mask_dir}")
    print(f"Overlay source:  {overlay_dir}")
    print(f"Export images:   {export_images}")
    print(f"Export masks:    {export_masks}")
    print(f"Export overlays: {export_overlays}")
    print(f"Target res:      {args.target_resolution}")
    print()

    records = []
    summary = {
        "exported": 0,
        "skipped_existing": 0,
        "missing_mask": 0,
        "read_error": 0,
    }

    size = (args.target_resolution, args.target_resolution)

    for fake_path in tqdm(fake_images, desc="Exportando", unit="img"):
        name = fake_path.name
        mask_path = mask_dir / name
        overlay_path = overlay_dir / name

        out_image = export_images / name
        out_mask = export_masks / name
        out_overlay = export_overlays / name

        if (
            not args.overwrite
            and out_image.exists()
            and out_mask.exists()
            and (not overlay_path.exists() or out_overlay.exists())
        ):
            summary["skipped_existing"] += 1
            records.append({"image": name, "status": "skipped_existing"})
            continue

        if not mask_path.exists():
            summary["missing_mask"] += 1
            records.append({"image": name, "status": "missing_mask"})
            continue

        fake = read_image(fake_path)
        mask = read_mask(mask_path)
        overlay = read_image(overlay_path) if overlay_path.exists() else None

        if fake is None or mask is None:
            summary["read_error"] += 1
            records.append({"image": name, "status": "read_error"})
            continue

        fake_256 = cv2.resize(fake, size, interpolation=cv2.INTER_LANCZOS4)
        mask_256 = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(out_image), fake_256)
        cv2.imwrite(str(out_mask), mask_256)

        if overlay is not None:
            overlay_256 = cv2.resize(overlay, size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(out_overlay), overlay_256)

        summary["exported"] += 1
        records.append({"image": name, "status": "exported"})

    payload = {
        "script": "03_exportar_faceswap_256.py",
        "updated_at": datetime.now().isoformat(),
        "workspace": str(workspace),
        "source_resolution": args.source_resolution,
        "target_resolution": args.target_resolution,
        "summary": summary,
        "detail": records,
    }
    write_log(log_path, payload)

    print()
    print("Resumen:")
    for key, value in summary.items():
        print(f"  {key:<20} {value}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
