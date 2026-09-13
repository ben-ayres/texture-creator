import io
import os
import uuid
from typing import Literal

import boto3
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat
from pydantic import BaseModel, Field

app = FastAPI(title="Patina Texture API", version="0.1.0")
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
async def create_job(file: UploadFile = File(...), material: str = "stone-walling", surface_height_m: float = 2.0, variant: str = "balanced", resolution: str = "HD"):
    if not os.getenv("R2_BUCKET_NAME"):
        raise HTTPException(status_code=503, detail="R2 storage is not configured yet.")
    payload = await file.read()
    job_id = str(uuid.uuid4())
    source_key = f"uploads/{job_id}/{file.filename or 'source-image'}"
    storage = r2_client()
    storage.put_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=source_key, Body=payload, ContentType=file.content_type or "image/jpeg")
    # The Modal worker is intentionally invoked by the deployment environment.
    # Until the worker is deployed, the job remains queued rather than pretending it completed.
    return {"job_id": job_id, "status": "queued", "source_key": source_key, "next": "Deploy services/texture-api/modal_worker.py to Modal to process this job."}