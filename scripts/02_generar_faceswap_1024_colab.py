# scripts/02_generar_faceswap_1024_colab.py
# =============================================================================
# Producción masiva de faceswaps en Colab/Drive.
#
# Flujo:
#   real_1024
#     -> fake_swap_1024
#     -> masks_swap_1024 = absdiff(real, fake) ∩ face_parsing(real)
#     -> overlays_swap_1024 para revisión visual
#
# Máscara facial:
#   Solo rostro interno. No incluye pelo, cuello, ropa ni fondo.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.face_parsing_mask import FaceParsingMasker  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class ProcessResult:
    image: str
    status: str
    source: str | None = None
    error: str | None = None
    fake_path: str | None = None
    mask_path: str | None = None
    overlay_path: str | None = None
    mask_pixels: int | None = None
    face_pixels: int | None = None
    elapsed_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera faceswaps 1024 con máscaras acotadas al rostro en Colab/Drive."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/content/drive/MyDrive/datasetpro_colab"),
        help="Raíz del workspace en Google Drive.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Máximo de imágenes objetivo a procesar.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolución maestra cuadrada. Recomendado: 1024.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para selección reproducible de rostro fuente.",
    )
    parser.add_argument(
        "--batch_log_every",
        type=int,
        default=25,
        help="Cada cuántas imágenes escribir log incremental.",
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="Tamaño de detección de InsightFace. 640 suele ser estable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerar aunque existan fake/mask/overlay.",
    )
    parser.add_argument(
        "--max_attempts_source",
        type=int,
        default=12,
        help="Intentos máximos para encontrar rostro fuente válido.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Intensidad visual de la máscara en overlay.",
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_rgb_sized(path: Path, resolution: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if h != resolution or w != resolution:
        image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    return image


def resolve_inswapper_path(workspace: Path) -> Path:
    candidates = [
        workspace / "models" / "insightface" / "inswapper_128.onnx",
        Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró inswapper_128.onnx. "
        f"Colócalo en {workspace / 'models' / 'insightface' / 'inswapper_128.onnx'}"
    )


def load_done_log(log_path: Path) -> dict[str, ProcessResult]:
    if not log_path.exists():
        return {}
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    detail = payload.get("detail", [])
    done: dict[str, ProcessResult] = {}
    for item in detail:
        if not isinstance(item, dict):
            continue
        image = item.get("image")
        if not image:
            continue
        done[image] = ProcessResult(
            image=image,
            status=item.get("status", "desconocido"),
            source=item.get("source"),
            error=item.get("error"),
            fake_path=item.get("fake_path"),
            mask_path=item.get("mask_path"),
            overlay_path=item.get("overlay_path"),
            mask_pixels=item.get("mask_pixels"),
            face_pixels=item.get("face_pixels"),
            elapsed_seconds=item.get("elapsed_seconds"),
        )
    return done


def write_log(
    log_path: Path,
    *,
    started_at: str,
    workspace: Path,
    resolution: int,
    seed: int,
    results: list[ProcessResult],
) -> None:
    ensure_dir(log_path.parent)
    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1

    payload = {
        "script": "02_generar_faceswap_1024_colab.py",
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(),
        "workspace": str(workspace),
        "resolution": resolution,
        "seed": seed,
        "summary": summary,
        "total_records": len(results),
        "detail": [asdict(result) for result in results],
    }
    tmp = log_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(log_path)


def build_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:, :, 2] = 255

    mask_bool = mask_gray > 0
    overlay[mask_bool] = cv2.addWeighted(
        image_bgr[mask_bool],
        1.0 - alpha,
        color_layer[mask_bool],
        alpha,
        0,
    )
    return overlay


def clean_binary_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    _, binary = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)

    # En 1024 conviene kernel proporcional. En 256, mantenerlo pequeño.
    kernel_size = max(5, int(round(resolution / 170)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def compute_restricted_mask(
    original_bgr: np.ndarray,
    fake_bgr: np.ndarray,
    face_masker: FaceParsingMasker,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    difference = cv2.absdiff(original_bgr, fake_bgr)
    diff_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    diff_mask = clean_binary_mask(diff_gray, resolution)

    face_mask = face_masker.predict_mask(original_bgr)
    if face_mask.shape[:2] != diff_mask.shape[:2]:
        face_mask = cv2.resize(
            face_mask,
            (diff_mask.shape[1], diff_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    final_mask = cv2.bitwise_and(diff_mask, face_mask)

    kernel_size = max(5, int(round(resolution / 200)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask, face_mask


def select_source_with_face(
    *,
    app,
    images: list[Path],
    target_path: Path,
    resolution: int,
    rng: random.Random,
    max_attempts: int,
) -> tuple[Path | None, np.ndarray | None, object | None]:
    candidates = [p for p in images if p != target_path]
    if not candidates:
        return None, None, None

    for _ in range(max_attempts):
        source_path = rng.choice(candidates)
        source_bgr = read_image_rgb_sized(source_path, resolution)
        if source_bgr is None:
            continue
        source_faces = app.get(source_bgr)
        if source_faces:
            return source_path, source_bgr, source_faces[0]

    return None, None, None


def main() -> int:
    args = parse_args()
    workspace = args.workspace
    data_dir = workspace / "data"
    models_dir = workspace / "models"

    real_dir = data_dir / "raw" / "real_1024"
    fake_dir = data_dir / "generated" / "fake_swap_1024"
    mask_dir = data_dir / "generated" / "masks_swap_1024"
    overlay_dir = data_dir / "generated" / "overlays_swap_1024"
    log_path = data_dir / "logs" / "faceswap_1024_log.json"

    for folder in [real_dir, fake_dir, mask_dir, overlay_dir, log_path.parent]:
        ensure_dir(folder)

    started_at = datetime.now().isoformat()
    rng = random.Random(args.seed)

    images = iter_images(real_dir)
    if not images:
        print(f"ERROR: No hay imágenes en {real_dir}")
        print("Coloca FFHQ 1024 en esa carpeta antes de ejecutar producción.")
        return 1

    targets = images[: args.limit]
    print("=" * 72)
    print("DATASETPRO — PRODUCCION FACESWAP 1024 COLAB")
    print("=" * 72)
    print(f"Workspace:       {workspace}")
    print(f"Real dir:        {real_dir}")
    print(f"Fake dir:        {fake_dir}")
    print(f"Mask dir:        {mask_dir}")
    print(f"Overlay dir:     {overlay_dir}")
    print(f"Targets:         {len(targets)} / {len(images)}")
    print(f"Resolution:      {args.resolution}")
    print(f"Overwrite:       {args.overwrite}")
    print()

    print("[1/4] Cargando modelos...")
    import insightface
    from insightface.app import FaceAnalysis

    inswapper_path = resolve_inswapper_path(workspace)
    print(f"  inswapper: {inswapper_path}")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    swapper = insightface.model_zoo.get_model(str(inswapper_path))

    face_masker = FaceParsingMasker(
        checkpoint_path=models_dir / "face_parsing" / "79999_iter.pth",
        include_hair=False,
    )
    print("  FaceAnalysis, inswapper y face parsing cargados.")
    print()

    existing = load_done_log(log_path)
    results = list(existing.values())
    by_image = {result.image: result for result in results}

    print("[2/4] Procesando imágenes...")
    processed_since_log = 0

    for target_path in tqdm(targets, desc="Faceswap", unit="img"):
        start = time.perf_counter()

        fake_path = fake_dir / target_path.name
        mask_path = mask_dir / target_path.name
        overlay_path = overlay_dir / target_path.name

        if (
            not args.overwrite
            and fake_path.exists()
            and mask_path.exists()
            and overlay_path.exists()
        ):
            result = ProcessResult(
                image=target_path.name,
                status="ya_existia",
                fake_path=str(fake_path),
                mask_path=str(mask_path),
                overlay_path=str(overlay_path),
                elapsed_seconds=0.0,
            )
            by_image[target_path.name] = result
            processed_since_log += 1
            if processed_since_log >= args.batch_log_every:
                write_log(
                    log_path,
                    started_at=started_at,
                    workspace=workspace,
                    resolution=args.resolution,
                    seed=args.seed,
                    results=list(by_image.values()),
                )
                processed_since_log = 0
            continue

        try:
            original_bgr = read_image_rgb_sized(target_path, args.resolution)
            if original_bgr is None:
                raise ValueError("No se pudo leer imagen objetivo")

            target_faces = app.get(original_bgr)
            if not target_faces:
                result = ProcessResult(
                    image=target_path.name,
                    status="sin_rostro_objetivo",
                    elapsed_seconds=time.perf_counter() - start,
                )
                by_image[target_path.name] = result
                processed_since_log += 1
                continue

            source_path, _source_bgr, source_face = select_source_with_face(
                app=app,
                images=images,
                target_path=target_path,
                resolution=args.resolution,
                rng=rng,
                max_attempts=args.max_attempts_source,
            )

            if source_path is None or source_face is None:
                result = ProcessResult(
                    image=target_path.name,
                    status="sin_rostro_fuente",
                    elapsed_seconds=time.perf_counter() - start,
                )
                by_image[target_path.name] = result
                processed_since_log += 1
                continue

            fake_bgr = original_bgr.copy()
            fake_bgr = swapper.get(
                fake_bgr,
                target_faces[0],
                source_face,
                paste_back=True,
            )

            final_mask, face_mask = compute_restricted_mask(
                original_bgr,
                fake_bgr,
                face_masker,
                args.resolution,
            )

            overlay = build_overlay(fake_bgr, final_mask, alpha=args.overlay_alpha)

            cv2.imwrite(str(fake_path), fake_bgr)
            cv2.imwrite(str(mask_path), final_mask)
            cv2.imwrite(str(overlay_path), overlay)

            result = ProcessResult(
                image=target_path.name,
                status="exitoso",
                source=source_path.name,
                fake_path=str(fake_path),
                mask_path=str(mask_path),
                overlay_path=str(overlay_path),
                mask_pixels=int(np.count_nonzero(final_mask)),
                face_pixels=int(np.count_nonzero(face_mask)),
                elapsed_seconds=time.perf_counter() - start,
            )
            by_image[target_path.name] = result

        except Exception as exc:
            result = ProcessResult(
                image=target_path.name,
                status="error",
                error=str(exc),
                elapsed_seconds=time.perf_counter() - start,
            )
            by_image[target_path.name] = result

        processed_since_log += 1
        if processed_since_log >= args.batch_log_every:
            write_log(
                log_path,
                started_at=started_at,
                workspace=workspace,
                resolution=args.resolution,
                seed=args.seed,
                results=list(by_image.values()),
            )
            processed_since_log = 0

    print()
    print("[3/4] Escribiendo log final...")
    final_results = list(by_image.values())
    write_log(
        log_path,
        started_at=started_at,
        workspace=workspace,
        resolution=args.resolution,
        seed=args.seed,
        results=final_results,
    )

    print("[4/4] Resumen...")
    summary: dict[str, int] = {}
    for result in final_results:
        summary[result.status] = summary.get(result.status, 0) + 1
    for status, count in sorted(summary.items()):
        print(f"  {status:<24} {count}")
    print(f"  log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
