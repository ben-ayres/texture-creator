from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).parent


def load(path: Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def save(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").save(path, quality=95)


source = load(ROOT / "floor-source.webp")
photoshop = load(ROOT / "photoshop-cropped.webp")
h, w = source.shape[:2]

# Stage 1: one complete reference paver gives a broad, low-risk correction.
stage1_src = np.float32([[980, 740], [1552, 780], [1544, 1340], [980, 1300]])
stage1_dst = np.float32([[980, 740], [1552, 740], [1552, 1300], [980, 1300]])
m1 = cv2.getPerspectiveTransform(stage1_src, stage1_dst)
stage1 = cv2.warpPerspective(source, m1, (w, h), borderMode=cv2.BORDER_REPLICATE)

# Stage 2: the user selects a normal rectangle in the mostly corrected image.
# The slight trapezoid represents several grout/paver references used together
# for fine correction, rather than relying on one tile again.
selected = np.float32([[180, 1110], [2840, 1110], [2810, 3330], [210, 3330]])
crop_w, crop_h = 2682, 2154
stage2_dst = np.float32([[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]])
m2 = cv2.getPerspectiveTransform(selected, stage2_dst)
stage2 = cv2.warpPerspective(stage1, m2, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE)
save(ROOT / "two-stage-corrected-crop.jpg", stage2)


def fit(array: np.ndarray, size=(670, 540)) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((size[0] - 20, size[1] - 45))
    canvas = Image.new("RGB", size, "#f8f5ef")
    canvas.paste(image, ((size[0] - image.width) // 2, 35))
    return canvas


items = [("Original", source), ("Stage 1: rough correction", stage1), ("Stage 2: refined crop", stage2), ("Photoshop reference", photoshop)]
sheet = Image.new("RGB", (1340, 1080), "#f8f5ef")
draw = ImageDraw.Draw(sheet)
for i, (label, array) in enumerate(items):
    x, y = (i % 2) * 670, (i // 2) * 540
    sheet.paste(fit(array), (x, y))
    draw.text((x + 12, y + 12), label, fill="#252723")

# Mark the selected area on the stage-1 panel.
stage1_panel = fit(stage1)
scale = min((670 - 20) / w, (540 - 45) / h)
ox = (670 - int(w * scale)) // 2
oy = 35 + (540 - 45 - int(h * scale)) // 2
points = [(int(ox + x * scale), int(oy + y * scale)) for x, y in selected]
ImageDraw.Draw(stage1_panel).line(points + [points[0]], fill="#b24b37", width=3)
sheet.paste(stage1_panel, (670, 0))
sheet.save(ROOT / "two-stage-perspective-comparison.jpg", quality=93)
print(f"stage2={stage2.shape[1]}x{stage2.shape[0]}")
print("Wrote two-stage-corrected-crop.jpg and two-stage-perspective-comparison.jpg")