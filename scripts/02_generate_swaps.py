# scripts/02_generate_swaps.py
"""
Genera pares (real, fake_swap) usando InsightFace buffalo_l.
Cada imagen real recibe un rostro fuente aleatorio del mismo dataset
para maximizar variación de identidad (§8.8.2 — mitigar sesgo de método).
"""
import cv2, random, insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from pathlib import Path
from tqdm import tqdm

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = get_model("inswapper_128.onnx", download=True)

REAL_DIR  = Path("data/raw/real")
FAKE_DIR  = Path("data/raw/fake_swap")
FAKE_DIR.mkdir(parents=True, exist_ok=True)

images = list(REAL_DIR.glob("*.png"))

for target_path in tqdm(images):
    source_path = random.choice(images)
    if source_path == target_path:
        continue
    target_img = cv2.imread(str(target_path))
    source_img = cv2.imread(str(source_path))
    target_faces = app.get(target_img)
    source_faces = app.get(source_img)
    if not target_faces or not source_faces:
        continue
    result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)
    cv2.imwrite(str(FAKE_DIR / target_path.name), result)