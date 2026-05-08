# scripts/04_generar_grid_qa_faceswap.py
# =============================================================================
# Genera grillas QA para revisión manual:
#   real | fake | mask | overlay
# =============================================================================

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera grillas QA para faceswap.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/content/drive/MyDrive/datasetpro_colab"),
        help="Raíz del workspace en Google Drive.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=64,
        help="Número máximo de muestras a incluir.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=180,
        help="Tamaño visual de cada celda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    parser.add_argument(
        "--use_export_256",
        action="store_true",
        help="Usar export_256 en vez de 1024 para grilla ligera.",
    )
    return parser.parse_args()


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_tile(path: Path, size: int, mode: str = "RGB") -> Image.Image:
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    img = img.resize((size, size), Image.Resampling.NEAREST if mode == "L" else Image.Resampling.LANCZOS)
    if mode == "L":
        img = img.convert("RGB")
    return img


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, label: str) -> None:
    draw.rectangle([x, y, x + w, y + 24], fill=(28, 28, 28))
    draw.text((x + 6, y + 6), label, fill=(255, 255, 255))


def main() -> int:
    args = parse_args()
    workspace = args.workspace
    data_dir = workspace / "data"
    qa_dir = workspace / "qa" / "grids"
    qa_dir.mkdir(parents=True, exist_ok=True)

    if args.use_export_256:
        real_dir = data_dir / "raw" / "real_1024"
        fake_dir = data_dir / "export_256" / "images"
        mask_dir = data_dir / "export_256" / "masks"
        overlay_dir = data_dir / "export_256" / "overlays"
        suffix = "export_256"
    else:
        real_dir = data_dir / "raw" / "real_1024"
        fake_dir = data_dir / "generated" / "fake_swap_1024"
        mask_dir = data_dir / "generated" / "masks_swap_1024"
        overlay_dir = data_dir / "generated" / "overlays_swap_1024"
        suffix = "master_1024"

    fake_images = iter_images(fake_dir)
    if not fake_images:
        print(f"ERROR: No hay imágenes fake en {fake_dir}")
        return 1

    complete = []
    for fake_path in fake_images:
        name = fake_path.name
        if (mask_dir / name).exists() and (overlay_dir / name).exists():
            complete.append(name)

    if not complete:
        print("ERROR: No hay muestras completas fake/mask/overlay.")
        return 1

    rng = random.Random(args.seed)
    sample_size = min(args.sample_size, len(complete))
    sample = rng.sample(complete, sample_size) if len(complete) > sample_size else complete

    columns = ["REAL", "FAKE", "MASK", "OVERLAY"]
    rows = len(sample)
    tile = args.tile_size
    gap = 6
    header = 28
    left_label = 150

    width = left_label + len(columns) * tile + (len(columns) - 1) * gap
    height = header + rows * tile + max(0, rows - 1) * gap

    grid = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(grid)

    for idx, column in enumerate(columns):
        x = left_label + idx * (tile + gap)
        draw_label(draw, x, 0, tile, column)

    for row_idx, name in enumerate(sample):
        y = header + row_idx * (tile + gap)
        draw.rectangle([0, y, left_label - 8, y + tile], fill=(28, 28, 28))
        draw.text((8, y + 8), name, fill=(230, 230, 230))

        paths = [
            real_dir / name,
            fake_dir / name,
            mask_dir / name,
            overlay_dir / name,
        ]

        for col_idx, path in enumerate(paths):
            x = left_label + col_idx * (tile + gap)
            if path.exists():
                mode = "L" if col_idx == 2 else "RGB"
                try:
                    img = open_tile(path, tile, mode=mode)
                    grid.paste(img, (x, y))
                except Exception:
                    draw.rectangle([x, y, x + tile, y + tile], fill=(80, 20, 20))
                    draw.text((x + 8, y + 8), "ERROR", fill=(255, 255, 255))
            else:
                draw.rectangle([x, y, x + tile, y + tile], fill=(50, 50, 50))
                draw.text((x + 8, y + 8), "MISSING", fill=(255, 255, 255))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = qa_dir / f"faceswap_qa_{suffix}_{timestamp}.png"
    grid.save(output)

    print(f"Grid QA generado: {output}")
    print(f"Muestras incluidas: {len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
