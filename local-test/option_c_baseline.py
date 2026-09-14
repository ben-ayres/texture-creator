from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).parent
SOURCE = ROOT / "floor-source.webp"
TARGET = ROOT / "target-texture.webp"


def load_rgb(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image)


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB").save(path, quality=94)


def flatten_lighting(rgb: np.ndarray) -> np.ndarray:
    # Remove broad lighting falloff without destroying the stone grain.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    light = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 75)
    median = float(np.median(light))
    lab[:, :, 0] = np.clip(lab[:, :, 0] * (median / np.maximum(light, 1)), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def make_edge_blended_repeat(rgb: np.ndarray, size: int = 1600) -> np.ndarray:
    tile = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    out = tile.astype(np.float32)
    band = max(24, size // 18)
    ramp = np.linspace(0, 1, band, dtype=np.float32)[None, :, None]
    # Blend only narrow opposing edges; do not mirror or rotate the source.
    out[:, :band] = out[:, :band] * (1 - ramp) + out[:, -band:] * ramp
    out[:, -band:] = out[:, -band:] * (1 - ramp) + out[:, :band] * ramp
    ramp_v = np.linspace(0, 1, band, dtype=np.float32)[:, None, None]
    out[:band] = out[:band] * (1 - ramp_v) + out[-band:] * ramp_v
    out[-band:] = out[-band:] * (1 - ramp_v) + out[:band] * ramp_v
    return np.clip(out, 0, 255).astype(np.uint8)


source = load_rgb(SOURCE)
target = load_rgb(TARGET)
h, w = source.shape[:2]

# Deliberately conservative local crop: avoids plant, table, and rug while
# retaining several complete pavers. No cloud/API call is made.
crop = source[int(h * 0.08):int(h * 0.72), int(w * 0.03):int(w * 0.70)]
crop = flatten_lighting(crop)
repeat = make_edge_blended_repeat(crop)

save_rgb(ROOT / "option-c-clean-crop.jpg", crop)
save_rgb(ROOT / "option-c-repeat.jpg", repeat)

thumb_w, thumb_h = 520, 520
sheet = Image.new("RGB", (thumb_w * 3, thumb_h), "white")
for i, (label, array) in enumerate(
    [("Original source", source), ("Conservative clean crop", crop), ("Local repeat baseline", repeat)]
):
    im = Image.fromarray(array).convert("RGB")
    im.thumbnail((thumb_w, thumb_h - 36))
    x = i * thumb_w + (thumb_w - im.width) // 2
    sheet.paste(im, (x, 30))
    ImageDraw.Draw(sheet).text((i * thumb_w + 12, 8), label, fill="black")
sheet.save(ROOT / "option-c-baseline-contact-sheet.jpg", quality=92)

print(f"source={w}x{h}; target={target.shape[1]}x{target.shape[0]}")
print("Wrote option-c-clean-crop.jpg, option-c-repeat.jpg, option-c-baseline-contact-sheet.jpg")