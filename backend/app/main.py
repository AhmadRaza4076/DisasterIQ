from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.config import settings
from app.schemas import AnalysisResult, BriefRequest, BriefResponse, DemoPair, ReportRequest
from app.services.cleanup import cleanup_old_jobs
from app.services.georef import enrich_zones_with_geo
from app.services.inference import list_demo_pairs, run_inference
from app.services.narrator import generate_brief
from app.services.report import build_field_report_pdf
from app.services.scoring import score_mask

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff"}
DEMO_PAIR_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

app = FastAPI(
    title="Disaster Damage Triage API",
    description="Satellite building damage assessment — Team DarkNem",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "inference_mode": settings.inference_mode,
        "demo_pairs": len(list_demo_pairs()),
    }


@app.get("/demo/pairs", response_model=list[DemoPair])
def demo_pairs() -> list[DemoPair]:
    return [DemoPair(**p) for p in list_demo_pairs()]


@app.get("/demo/images/{filename}")
def demo_image(filename: str) -> FileResponse:
    if Path(filename).name != filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if any(sep in filename for sep in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = settings.demo_data_dir / "images" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


def _validate_upload(upload: UploadFile) -> None:
    if upload.content_type and upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {upload.content_type}",
        )
    if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 25 MB upload limit")


def _save_upload(upload: UploadFile, dest: Path, job_dir: Path) -> None:
    """Stream an upload to disk, aborting (and cleaning up) if it exceeds the size cap."""
    written = 0
    chunk_size = 1024 * 1024
    try:
        with dest.open("wb") as f:
            while chunk := upload.file.read(chunk_size):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=400, detail="Image exceeds 25 MB upload limit"
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    try:
        with Image.open(dest) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def _run_analysis_pipeline(
    pre_path: Path,
    post_path: Path,
    out_dir: Path,
    demo_pair_id: str | None,
) -> AnalysisResult:
    """Synchronous inference + scoring + geo enrichment (run off the event loop)."""
    mask_path, mode, confidence_path = run_inference(pre_path, post_path, out_dir)
    result = score_mask(
        mask_path,
        grid_rows=settings.grid_rows,
        grid_cols=settings.grid_cols,
        confidence_path=confidence_path,
    )
    result.inference_mode = mode
    result.pair_id = demo_pair_id or pre_path.stem.replace("_pre_disaster", "")

    if demo_pair_id:
        label_path = settings.demo_data_dir / "labels" / f"{demo_pair_id}_post_disaster.json"
        result.geo_available = enrich_zones_with_geo(result.zones, label_path)

    return result


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    pre_image: UploadFile | None = File(None),
    post_image: UploadFile | None = File(None),
    demo_pair_id: str | None = Form(None),
) -> AnalysisResult:
    cleanup_old_jobs(settings.upload_dir, settings.output_dir)
    job_id = uuid.uuid4().hex
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir = settings.output_dir / job_id

    if demo_pair_id:
        if not DEMO_PAIR_ID_RE.match(demo_pair_id):
            raise HTTPException(status_code=400, detail="Invalid demo_pair_id")
        demo_img = settings.demo_data_dir / "images"
        pre_path = demo_img / f"{demo_pair_id}_pre_disaster.png"
        post_path = demo_img / f"{demo_pair_id}_post_disaster.png"
        if not pre_path.exists() or not post_path.exists():
            raise HTTPException(status_code=404, detail=f"Demo pair not found: {demo_pair_id}")
    else:
        if pre_image is None or post_image is None:
            raise HTTPException(
                status_code=400,
                detail="Provide pre_image and post_image uploads, or demo_pair_id",
            )
        _validate_upload(pre_image)
        _validate_upload(post_image)
        pre_name = Path(pre_image.filename or "pre.png").name or "pre.png"
        post_name = Path(post_image.filename or "post.png").name or "post.png"
        pre_path = job_dir / pre_name
        post_path = job_dir / post_name
        _save_upload(pre_image, pre_path, job_dir)
        _save_upload(post_image, post_path, job_dir)

    try:
        return await anyio.to_thread.run_sync(
            _run_analysis_pipeline, pre_path, post_path, out_dir, demo_pair_id
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/brief", response_model=BriefResponse)
async def brief(body: BriefRequest) -> BriefResponse:
    return await generate_brief(body.analysis, body.context)


@app.post("/report/pdf")
async def report_pdf(body: ReportRequest) -> Response:
    pair_id = body.analysis.get("pair_id")
    pdf_bytes = await anyio.to_thread.run_sync(
        lambda: build_field_report_pdf(body.analysis, body.brief, pair_id=pair_id)
    )
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", str(pair_id)) if pair_id else ""
    filename = f"disasteriq-report-{safe_id or 'analysis'}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/analyze-and-brief")
async def analyze_and_brief(
    pre_image: UploadFile | None = File(None),
    post_image: UploadFile | None = File(None),
    demo_pair_id: str | None = Form(None),
    context: str | None = Form(None),
) -> dict:
    analysis = await analyze(pre_image=pre_image, post_image=post_image, demo_pair_id=demo_pair_id)
    brief_resp = await generate_brief(analysis.model_dump(exclude={"mask_base64"}), context)
    return {"analysis": analysis, "brief": brief_resp}
