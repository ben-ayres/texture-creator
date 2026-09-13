"""First GPU-worker scaffold for the real albedo prototype.

Deploy with the Modal CLI after the API and sample pipeline are reviewed:
    modal deploy modal_worker.py

The worker deliberately does not invent a finished texture yet. It provides the
deployment boundary where perspective correction, illumination estimation,
stone/grout segmentation, and seam synthesis will be tested against Patina's
reference images.
"""

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "Pillow==11.1.0", "numpy==2.2.3", "opencv-python-headless==4.11.0.86"
)
app = modal.App("patina-texture-worker")


@app.function(image=image, timeout=900, cpu=4, memory=8192)
def process_albedo(source_bytes: bytes, surface_height_m: float, variant: str = "balanced", resolution: str = "HD") -> dict:
    """Placeholder boundary for the validated albedo pipeline.

    The first implementation should be benchmarked against the supplied stone
    wall and paver images before enabling production downloads.
    """
    return {
        "status": "worker-ready",
        "message": "GPU worker connected; albedo pipeline implementation is the next processing milestone.",
        "surface_height_m": surface_height_m,
        "variant": variant,
        "resolution": resolution,
        "source_bytes": len(source_bytes),
    }