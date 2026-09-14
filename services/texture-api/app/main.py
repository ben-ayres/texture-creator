import io
import json
import logging
import os
import uuid
from typing import Literal

import boto3
import modal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat
from pydantic import BaseModel, Field

app = FastAPI(title="Patina Texture API", version="0.1.0")
logger = logging.getLogger("patina.texture")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResult(BaseModel):
    suitable: bool
    filename: str
    width: int
    height: int
    aspect_ratio: float
    megapixels: float
    mean_brightness: float
    warnings: list[str] = Field(default_factory=list)
    message: str


class TextureJob(BaseModel):
    material: Literal["stone-walling", "stone-pavers"] = "stone-walling"
    surface_height_m: float = Field(gt=0, le=20)
    variant: Literal["original", "balanced", "reduced"] = "balanced"
    grout_style: str = "preserve-existing"
    grout_tone: Literal["cool", "warm"] = "warm"
    resolution: Literal["HD", "1K", "2K"] = "HD"


def r2_client():
    required = ["R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise HTTPException(status_code=503, detail=f"Storage is not configured: {', '.join(missing)}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def signed_result(storage, key: str) -> str:
    return storage.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["R2_BUCKET_NAME"], "Key": key},
        ExpiresIn=3600,
    )


async def parse_corners(corners: str) -> list[list[float]] | None:
    try:
        points = json.loads(corners) if corners else None
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Surface corners must be valid JSON.") from exc
    if points is not None and (
        not isinstance(points, list) or len(points) != 4 or any(
            not isinstance(point, list) or len(point) != 2
            or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in point)
            for point in points
        )
    ):
        raise HTTPException(status_code=422, detail="Surface corners must contain four normalised [x, y] points.")
    return points


@app.get("/health")
def health():
    return {"ok": True, "service": "patina-texture-api", "storage_configured": bool(os.getenv("R2_BUCKET_NAME"))}


@app.post("/v1/analyze", response_model=AnalyzeResult)
async def analyze_source(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WebP image.")
    payload = await file.read()
    if len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 25 MB.")
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc
    width, height = image.size
    brightness = sum(ImageStat.Stat(image).mean) / 3
    warnings: list[str] = []
    if min(width, height) < 800:
        warnings.append("Low source resolution; fine surface detail may be limited.")
    if brightness < 35 or brightness > 225:
        warnings.append("Exposure may limit reliable lighting correction.")
    if width / max(height, 1) > 4.5 or height / max(width, 1) > 4.5:
        warnings.append("Very wide or tall framing; select a smaller material area.")
    suitable = width >= 800 and height >= 600 and not (brightness < 15 or brightness > 245)
    return AnalyzeResult(
        suitable=suitable,
        filename=file.filename or "upload",
        width=width,
        height=height,
        aspect_ratio=round(width / max(height, 1), 3),
        megapixels=round(width * height / 1_000_000, 2),
        mean_brightness=round(brightness, 1),
        warnings=warnings,
        message="Suitable source image." if suitable else "Please use a clearer, better-exposed source image.",
    )


@app.post("/v1/jobs")
async def create_job(file: UploadFile = File(...), material: str = "stone-walling", surface_height_m: float = 2.0, variant: str = "balanced", resolution: str = "HD", corners: str = "", flattened_key: str = ""):
    if not os.getenv("R2_BUCKET_NAME"):
        raise HTTPException(status_code=503, detail="R2 storage is not configured yet.")
    payload = await file.read()
    job_id = str(uuid.uuid4())
    source_key = f"uploads/{job_id}/{file.filename or 'source-image'}"
    storage = r2_client()
    if flattened_key:
        try:
            payload = storage.get_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=flattened_key)["Body"].read()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="The flattened surface could not be found. Please flatten the surface again.") from exc
    storage.put_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=source_key, Body=payload, ContentType=file.content_type or "image/jpeg")
    corner_points = await parse_corners(corners)
    if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
        return {"job_id": job_id, "status": "stored", "source_key": source_key, "next": "Add Modal credentials and deploy the worker."}
    try:
        worker = modal.Function.from_name("patina-texture-worker", "process_albedo")
        result_bytes = await worker.remote.aio(payload, surface_height_m, variant, resolution, corner_points, material)
        result_key = f"processed/{job_id}/albedo-{resolution.lower()}.jpg"
        storage.put_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=result_key, Body=result_bytes, ContentType="image/jpeg")
        return {
            "job_id": job_id,
            "status": "complete",
            "source_key": source_key,
            "result_key": result_key,
            "result_url": signed_result(storage, result_key),
            "message": "Lighting-normalised seamless albedo prototype created.",
        }
    except Exception as exc:
        logger.exception("Modal worker failed for job %s", job_id)
        return {
            "job_id": job_id,
            "status": "queued",
            "source_key": source_key,
            "message": "The image was saved, but Modal could not process it yet. Check Render logs for the detailed reason.",
            "worker_error": str(exc),
        }


@app.post("/v1/flatten")
async def flatten_surface(file: UploadFile = File(...), corners: str = ""):
    """Perspective-correct the selected surface and return it for review."""
    if not os.getenv("R2_BUCKET_NAME"):
        raise HTTPException(status_code=503, detail="R2 storage is not configured yet.")
    payload = await file.read()
    corner_points = await parse_corners(corners)
    job_id = str(uuid.uuid4())
    storage = r2_client()
    source_key = f"flattened/{job_id}/surface.jpg"
    if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
        raise HTTPException(status_code=503, detail="Modal processing is not configured yet.")
    try:
        worker = modal.Function.from_name("patina-texture-worker", "process_flatten")
        result_bytes = await worker.remote.aio(payload, corner_points)
        storage.put_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=source_key, Body=result_bytes, ContentType="image/jpeg")
        return {"status": "complete", "result_key": source_key, "result_url": signed_result(storage, source_key), "message": "Surface flattened for review."}
    except Exception as exc:
        logger.exception("Modal flatten failed for job %s", job_id)
        raise HTTPException(status_code=502, detail=f"Surface flattening failed: {exc}") from exc