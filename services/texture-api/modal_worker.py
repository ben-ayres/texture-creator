"""Modal worker for Patina's first real albedo prototype.

This first pass preserves source pixels, normalises broad illumination, and
builds a non-mirrored tile from the selected surface. It is a processing
prototype, not the final stone/grout segmentation model.
"""

import io
import modal
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "Pillow==11.1.0", "numpy==2.2.3", "opencv-python-headless==4.11.0.86"
)
app = modal.App("patina-texture-worker")


RESOLUTIONS = {"HD": 1080, "1K": 1024, "2K": 2048}


def _illumination_normalise(image: Image.Image) -> Image.Image:
    """Remove broad light falloff while retaining small surface detail."""
    rgb = image.convert("RGB")
    luminance = rgb.convert("L").filter(ImageFilter.GaussianBlur(max(12, min(rgb.size) // 10)))
    mean = ImageStat.Stat(luminance).mean[0]
    target = Image.new("L", luminance.size, int(max(1, min(255, mean))))
    ratio = ImageChops.subtract(target, luminance, scale=1.0, offset=128)
    corrected = ImageChops.add(rgb, Image.merge("RGB", (ratio, ratio, ratio)), scale=1.0, offset=-128)
    return ImageEnhance.Contrast(corrected).enhance(0.98)


def _perspective_correct(image: Image.Image, corners: list[list[float]] | None) -> Image.Image:
    if not corners or len(corners) != 4:
        return image
    source = np.array(image.convert("RGB"))
    points = np.array(corners, dtype=np.float32)
    # Points are normalised [x, y] values in clockwise order.
    points[:, 0] *= source.shape[1]
    points[:, 1] *= source.shape[0]
    width = max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))
    # Keep the selected surface's aspect ratio. The old implementation later
    # forced this result into a square, which stretched flooring joints and
    # made a correct projective warp look curved.
    output_width = max(256, int(width))
    output_height = max(256, int(height))
    destination = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, destination)
    warped = cv2.warpPerspective(source, matrix, (output_width, output_height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _edge_blend(tile: np.ndarray, band: int) -> np.ndarray:
    """Blend opposing borders so the tile joins without mirrored geometry."""
    result = tile.astype(np.float32)
    height, width = result.shape[:2]
    band = max(8, min(band, width // 4, height // 4))
    for distance in range(band):
        weight = (distance + 1) / (band + 1)
        left, right = result[:, distance].copy(), result[:, width - band + distance].copy()
        mix = left * (1 - weight) + right * weight
        result[:, distance] = mix
        result[:, width - band + distance] = mix
        top, bottom = result[distance, :].copy(), result[height - band + distance, :].copy()
        mix = top * (1 - weight) + bottom * weight
        result[distance, :] = mix
        result[height - band + distance, :] = mix
    return np.clip(result, 0, 255).astype(np.uint8)


def _seamless_tile(image: Image.Image, resolution: int, variant: str, material: str) -> Image.Image:
    """Build a rectangular, non-mirrored repeat without changing surface geometry."""
    source_width, source_height = image.size
    scale = resolution / max(source_width, source_height)
    target_width = max(256, round(source_width * scale))
    target_height = max(256, round(source_height * scale))
    crop = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    pixels = np.array(crop.convert("RGB"))
    # Blend opposing borders while retaining the selected surface aspect ratio.
    band = min(target_width, target_height) // (10 if material == "stone-pavers" else 14)
    result = Image.fromarray(_edge_blend(pixels, band))
    if variant == "reduced":
        result = ImageEnhance.Color(result).enhance(0.96)
    elif variant == "original":
        result = ImageEnhance.Color(result).enhance(1.02)
    return result


@app.function(image=image, timeout=900, cpu=4, memory=8192)
def process_albedo(source_bytes: bytes, surface_height_m: float, variant: str = "balanced", resolution: str = "HD", corners: list[list[float]] | None = None, material: str = "stone-walling") -> bytes:
    """Return a conservative, lighting-normalised, seamless albedo JPEG."""
    source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    output_size = RESOLUTIONS.get(resolution, 1080)
    corrected = _illumination_normalise(_perspective_correct(source, corners))
    result = _seamless_tile(corrected, output_size, variant, material)
    output = io.BytesIO()
    result.save(output, format="JPEG", quality=95, subsampling=0, optimize=True)
    return output.getvalue()