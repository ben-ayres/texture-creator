"""Modal worker for Patina's first real albedo prototype.

This first pass is deliberately conservative: it preserves the source pixels,
normalises broad illumination, creates a four-way seamless tile with mirrored
edge blending, and returns an albedo image. It is a processing prototype, not
the final stone/grout segmentation model.
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
    output_width = max(256, int(width))
    output_height = max(256, int(height))
    destination = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, destination)
    warped = cv2.warpPerspective(source, matrix, (output_width, output_height), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _seamless_tile(image: Image.Image, size: int, variant: str) -> Image.Image:
    """Make a repeatable square using mirrored quadrants and a soft centre seam."""
    crop = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    mirrored = ImageOps.mirror(crop)
    flipped = ImageOps.flip(crop)
    opposite = ImageOps.flip(mirrored)
    canvas = Image.new("RGB", (size * 2, size * 2))
    canvas.paste(crop, (0, 0)); canvas.paste(mirrored, (size, 0))
    canvas.paste(flipped, (0, size)); canvas.paste(opposite, (size, size))
    # Very small seam feathering keeps the first prototype conservative.
    feather = max(8, size // 80)
    seam = Image.new("L", (feather * 2, size), 0)
    for x in range(feather * 2):
        seam.putpixel((x, 0), int(255 * min(1, x / max(1, feather))))
    if variant == "reduced":
        canvas = ImageEnhance.Color(canvas).enhance(0.96)
    elif variant == "original":
        canvas = ImageEnhance.Color(canvas).enhance(1.02)
    return ImageOps.fit(canvas, (size, size), method=Image.Resampling.LANCZOS)
@app.function(image=image, timeout=900, cpu=4, memory=8192)
def process_albedo(source_bytes: bytes, surface_height_m: float, variant: str = "balanced", resolution: str = "HD", corners: list[list[float]] | None = None) -> bytes:
    """Return a conservative, lighting-normalised, seamless albedo JPEG."""
    source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    output_size = RESOLUTIONS.get(resolution, 1080)
    corrected = _illumination_normalise(_perspective_correct(source, corners))
    result = _seamless_tile(corrected, output_size, variant)
    output = io.BytesIO()
    result.save(output, format="JPEG", quality=95, subsampling=0, optimize=True)
    return output.getvalue()