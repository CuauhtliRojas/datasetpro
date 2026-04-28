# scripts/03_generate_inpainting.py
"""
Genera fakes de tipo edición local usando SD-inpainting sobre región facial.
Cubre la tipología 'edición' de §8.1.1 y aporta inconsistencias fotométricas (§8.7.1).
"""
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np, cv2
from pathlib import Path
from tqdm import tqdm

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

REAL_DIR   = Path("data/raw/real")
INPAINT_DIR = Path("data/raw/fake_inpaint")
INPAINT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "a different person's face, photorealistic",
    "slightly altered facial features, photorealistic",
    "face with different skin tone, photorealistic",
]

def get_face_mask(img_np):
    """Máscara de la región facial usando haar cascade — zona de interés (§8.1.3)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.1, 4)
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 255
    return mask

for img_path in tqdm(list(REAL_DIR.glob("*.png"))[:12500]):
    img = Image.open(img_path).convert("RGB").resize((512, 512))
    img_np = np.array(img)
    face_mask = get_face_mask(img_np)
    if face_mask.sum() == 0:
        continue
    mask_pil = Image.fromarray(face_mask).resize((512, 512))
    prompt = np.random.choice(PROMPTS)
    result = pipe(prompt=prompt, image=img, mask_image=mask_pil,
                  num_inference_steps=30, guidance_scale=7.5).images[0]
    result.save(INPAINT_DIR / img_path.name)