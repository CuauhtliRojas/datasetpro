# src/datasetpro/face_parsing_mask.py

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
VENDOR_FACE_PARSING = ROOT / "vendor" / "face_parsing_pytorch"

MODEL_FILE = VENDOR_FACE_PARSING / "model.py"
RESNET_FILE = VENDOR_FACE_PARSING / "resnet.py"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró model.py en {MODEL_FILE}. "
        "Copia face-parsing.PyTorch/model.py a vendor/face_parsing_pytorch/model.py"
    )

if not RESNET_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró resnet.py en {RESNET_FILE}. "
        "Copia face-parsing.PyTorch/resnet.py a vendor/face_parsing_pytorch/resnet.py"
    )

if str(VENDOR_FACE_PARSING) not in sys.path:
    sys.path.insert(0, str(VENDOR_FACE_PARSING))

from model import BiSeNet  # noqa: E402


class FaceParsingMasker:
    """
    Genera una máscara binaria de región facial usando face-parsing.PyTorch.

    Salida:
        np.ndarray uint8 con valores 0 o 255.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "models/face_parsing/79999_iter.pth",
        device: str | None = None,
        include_hair: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.include_hair = include_hair

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint de face parsing: {self.checkpoint_path}"
            )

        self.net = BiSeNet(n_classes=19)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Recibe imagen BGR de OpenCV.
        Devuelve máscara binaria del rostro en tamaño original.
        """
        if image_bgr is None:
            raise ValueError("image_bgr no puede ser None")

        h, w = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        resized = pil_image.resize((512, 512), Image.BILINEAR)

        tensor = self.to_tensor(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.net(tensor)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0).astype(np.uint8)

        # Clases CelebAMask-HQ comunes en face-parsing.PyTorch:
        # 0 background
        # 1 skin
        # 2 l_brow
        # 3 r_brow
        # 4 l_eye
        # 5 r_eye
        # 6 eye_g
        # 7 l_ear
        # 8 r_ear
        # 9 ear_r
        # 10 nose
        # 11 mouth
        # 12 u_lip
        # 13 l_lip
        # 14 neck
        # 15 neck_l
        # 16 cloth
        # 17 hair
        # 18 hat

        face_classes = [1, 2, 3, 4, 5, 10, 11, 12, 13]

        if self.include_hair:
            face_classes.append(17)

        mask = np.isin(parsing, face_classes).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask