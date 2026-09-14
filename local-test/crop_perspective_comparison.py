from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).parent


def load(path: Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def save(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8), "RGB").save(path, quality=95)


source = load(ROOT / "floor-source.webp")
photoshop_full = load(ROOT / "photoshop-full.webp")
photoshop_crop = load(ROOT / "photoshop-cropped.webp")
target = load(ROOT / "target-texture.webp")

# Local approximation of the crop boundary visible in the user's Photoshop
# example. This is explicit and local: no AI or cloud service is used.
h, w = source.shape[:2]
corners = np.float32([
    [0.04 * w, 0.08 * h],
    [0.89 * w, 0.08 * h],
    [0.87 * w, 0.65 * h],
    [0.05 * w, 0.65 * h],
])
out_w, out_h = photoshop_crop.shape[1], photoshop_crop.shape[0]
destination = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
matrix = cv2.getPerspectiveTransform(corners, destination)
local_front_on = cv2.warpPerspective(source, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)
save(ROOT / "local-front-on-crop.jpg", local_front_on)

# A non-mirrored edge-blended repeat of the local result.
repeat = cv2.resize(local_front_on, (1600, 1285), interpolation=cv2.INTER_AREA)
band = 65
result = repeat.astype(np.float32)
for i in range(band):
    weight = (i + 1) / (band + 1)
    left, right = result[:, i].copy(), result[:, -band + i].copy()
    mix = left * (1 - weight) + right * weight
    result[:, i], result[:, -band + i] = mix, mix
    top, bottom = result[i, :].copy(), result[-band + i, :].copy()
    mix = top * (1 - weight) + bottom * weight
    result[i, :], result[-band + i, :] = mix, mix
save(ROOT / "local-front-on-repeat.jpg", np.clip(result, 0, 255))


def panel(array: np.ndarray, width: int = 430, height: int = 350) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((width - 20, height - 45))
    canvas = Image.new("RGB", (width, height), "#f8f5ef")
    canvas.paste(image, ((width - image.width) // 2, 34))
    return canvas


items = [
    ("Original", source),
    ("Photoshop full edit", photoshop_full),
    ("Photoshop cropped target", photoshop_crop),
    ("Local crop + perspective", local_front_on),
    ("Local repeat", result.astype(np.uint8)),
    ("Reference texture", target),
]
sheet = Image.new("RGB", (430 * 3, 350 * 2), "#f8f5ef")
draw = ImageDraw.Draw(sheet)
for index, (label, array) in enumerate(items):
    x, y = (index % 3) * 430, (index // 3) * 350
    sheet.paste(panel(array), (x, y))
    draw.text((x + 12, y + 10), label, fill="#252723")
sheet.save(ROOT / "crop-perspective-comparison.jpg", quality=93)
print("Wrote local-front-on-crop.jpg, local-front-on-repeat.jpg, crop-perspective-comparison.jpg")