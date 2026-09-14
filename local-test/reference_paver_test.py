from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).parent


def load(path: Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def save(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB").save(path, quality=94)


source = load(ROOT / "floor-source.webp")
h, w = source.shape[:2]

# Four corners of one complete paver selected from the local 756px preview,
# scaled back to the 3024x4032 original.
reference = np.float32([[980, 740], [1552, 780], [1544, 1340], [980, 1300]])
destination = np.float32([[980, 740], [1552, 740], [1552, 1300], [980, 1300]])
matrix = cv2.getPerspectiveTransform(reference, destination)
rectified = cv2.warpPerspective(source, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
save(ROOT / "reference-paver-rectified-full.jpg", rectified)

# Crop the same broad usable area as the Photoshop example, but after the
# full-image rectification.
crop_box = np.float32([[[192, 1148], [2816, 1148], [2816, 3324], [192, 3324]]])
mapped = cv2.perspectiveTransform(crop_box, matrix)[0]
left, top = np.floor(mapped.min(axis=0)).astype(int)
right, bottom = np.ceil(mapped.max(axis=0)).astype(int)
crop = rectified[max(0, top):min(h, bottom), max(0, left):min(w, right)]
save(ROOT / "reference-paver-rectified-crop.jpg", crop)


def thumb(array: np.ndarray, size=(430, 350)) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((size[0] - 20, size[1] - 40))
    canvas = Image.new("RGB", size, "#f8f5ef")
    canvas.paste(image, ((size[0] - image.width) // 2, 32))
    return canvas


items = [("Original", source), ("Rectified full image", rectified), ("Rectified texture crop", crop), ("Photoshop crop", load(ROOT / "photoshop-cropped.webp"))]
sheet = Image.new("RGB", (860, 700), "#f8f5ef")
draw = ImageDraw.Draw(sheet)
for i, (label, array) in enumerate(items):
    x, y = (i % 2) * 430, (i // 2) * 350
    sheet.paste(thumb(array), (x, y))
    draw.text((x + 12, y + 10), label, fill="#252723")

# Mark the reference paver on the original panel.
overlay = thumb(source)
overlay_draw = ImageDraw.Draw(overlay)
scale = min((430 - 20) / w, (350 - 40) / h)
offset_x = (430 - int(w * scale)) // 2
offset_y = 32 + (350 - 40 - int(h * scale)) // 2
points = [(int(offset_x + x * scale), int(offset_y + y * scale)) for x, y in reference]
overlay_draw.line(points + [points[0]], fill="#b24b37", width=3)
sheet.paste(overlay, (0, 0))
sheet.save(ROOT / "reference-paver-comparison.jpg", quality=93)
print(f"source={w}x{h}; rectified crop={crop.shape[1]}x{crop.shape[0]}")
print("Wrote reference-paver-rectified-full.jpg, reference-paver-rectified-crop.jpg, reference-paver-comparison.jpg")