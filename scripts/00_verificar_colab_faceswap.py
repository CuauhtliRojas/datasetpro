# scripts/00_verificar_colab_faceswap.py
# =============================================================================
# Verificación de entorno para producción Colab/Drive de faceswap 1024.
# =============================================================================

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica estructura, dependencias y archivos requeridos para faceswap en Colab."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/content/drive/MyDrive/datasetpro_colab"),
        help="Raíz del workspace en Google Drive.",
    )
    return parser.parse_args()


def check_module(module_name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, "NO INSTALADO"
    return True, "OK"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    total = 0
    for pattern in patterns:
        total += len(list(path.glob(pattern)))
    return total


def main() -> int:
    args = parse_args()
    workspace = args.workspace

    repo_dir = workspace / "repo" / "datasetpro"
    data_dir = workspace / "data"
    models_dir = workspace / "models"
    qa_dir = workspace / "qa"

    expected_dirs = [
        workspace,
        repo_dir,
        data_dir / "raw" / "real_1024",
        data_dir / "generated" / "fake_swap_1024",
        data_dir / "generated" / "masks_swap_1024",
        data_dir / "generated" / "overlays_swap_1024",
        data_dir / "export_256" / "images",
        data_dir / "export_256" / "masks",
        data_dir / "export_256" / "overlays",
        data_dir / "logs",
        models_dir / "insightface",
        models_dir / "face_parsing",
        qa_dir / "grids",
        qa_dir / "reports",
    ]

    print("=" * 72)
    print("DATASETPRO — VERIFICACION COLAB FACESWAP")
    print("=" * 72)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print(f"Workspace:  {workspace}")
    print(f"Repo dir:   {repo_dir}")
    print()

    print("[1/5] Creando/verificando estructura de carpetas...")
    for path in expected_dirs:
        ensure_dir(path)
        print(f"  OK  {path}")
    print()

    print("[2/5] Verificando dependencias Python principales...")
    modules = [
        "cv2",
        "numpy",
        "PIL",
        "torch",
        "torchvision",
        "insightface",
        "onnxruntime",
        "tqdm",
    ]

    missing = []
    for module_name in modules:
        ok, status = check_module(module_name)
        print(f"  {module_name:<16} {status}")
        if not ok:
            missing.append(module_name)
    print()

    print("[3/5] Verificando CUDA / Torch...")
    try:
        import torch

        print(f"  torch.__version__:       {torch.__version__}")
        print(f"  cuda disponible:         {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda device count:       {torch.cuda.device_count()}")
            print(f"  cuda device name:        {torch.cuda.get_device_name(0)}")
            print(f"  cuda capability:         {torch.cuda.get_device_capability(0)}")
        else:
            print("  ADVERTENCIA: CUDA no está disponible. Faceswap masivo será lento o inviable.")
    except Exception as exc:
        print(f"  ERROR importando torch: {exc}")
        missing.append("torch")
    print()

    print("[4/5] Verificando modelos externos requeridos...")
    inswapper_candidates = [
        models_dir / "insightface" / "inswapper_128.onnx",
        Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
    ]
    face_parsing_checkpoint = models_dir / "face_parsing" / "79999_iter.pth"

    found_inswapper = None
    for candidate in inswapper_candidates:
        if candidate.exists():
            found_inswapper = candidate
            break

    if found_inswapper:
        print(f"  OK  inswapper_128.onnx: {found_inswapper}")
    else:
        print("  FALTA inswapper_128.onnx")
        print(f"       Ruta recomendada: {models_dir / 'insightface' / 'inswapper_128.onnx'}")

    if face_parsing_checkpoint.exists():
        print(f"  OK  face parsing checkpoint: {face_parsing_checkpoint}")
    else:
        print("  FALTA face parsing checkpoint 79999_iter.pth")
        print(f"       Ruta recomendada: {face_parsing_checkpoint}")
    print()

    print("[5/5] Conteos actuales...")
    folders = {
        "real_1024": data_dir / "raw" / "real_1024",
        "fake_swap_1024": data_dir / "generated" / "fake_swap_1024",
        "masks_swap_1024": data_dir / "generated" / "masks_swap_1024",
        "overlays_swap_1024": data_dir / "generated" / "overlays_swap_1024",
        "export_256/images": data_dir / "export_256" / "images",
        "export_256/masks": data_dir / "export_256" / "masks",
        "export_256/overlays": data_dir / "export_256" / "overlays",
    }

    for label, folder in folders.items():
        print(f"  {label:<24} {count_images(folder)}")
    print()

    print("=" * 72)
    if missing or not found_inswapper or not face_parsing_checkpoint.exists():
        print("RESULTADO: entorno incompleto.")
        print()
        if missing:
            print("Dependencias faltantes:")
            for module_name in missing:
                print(f"  - {module_name}")
        if not found_inswapper:
            print("Archivo faltante:")
            print(f"  - {models_dir / 'insightface' / 'inswapper_128.onnx'}")
        if not face_parsing_checkpoint.exists():
            print("Archivo faltante:")
            print(f"  - {face_parsing_checkpoint}")
        print()
        print("Puedes crear carpetas con este script, pero antes de producir necesitas resolver faltantes.")
        return 1

    print("RESULTADO: entorno listo para producción faceswap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
