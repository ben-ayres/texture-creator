# Patina texture API

This is the Render-side API for the first real albedo prototype.

## Local run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

## Endpoints

- `GET /health` — service and storage check
- `POST /v1/analyze` — validates an uploaded source image and returns suitability metadata
- `POST /v1/jobs` — stores the source image in R2 and queues a processing job

## Render setup

Create a Render Web Service using this directory as the root directory. Use:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Add the variables from `.env.example` in Render's Environment page. Never commit
the real values to GitHub.

## Modal setup

After reviewing the worker boundary and testing the API, deploy the worker from
the project root with `modal deploy services/texture-api/modal_worker.py`.
The actual perspective, illumination, segmentation, and seam-generation stages
must be benchmarked against Patina's reference image set before enabling final
downloads.