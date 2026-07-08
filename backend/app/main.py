from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.config import settings
from app.schemas import AnalysisResult, BriefRequest, BriefResponse, DemoPair, ReportRequest
from app.services.georef import enrich_zones_with_geo
from app.services.inference import list_demo_pairs, run_inference
from app.services.narrator import generate_brief
from app.services.report import build_field_report_pdf
from app.services.scoring import score_mask

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff"}

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


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    pre_image: UploadFile | None = File(None),
    post_image: UploadFile | None = File(None),
    demo_pair_id: str | None = Form(None),
) -> AnalysisResult:
    job_id = uuid.uuid4().hex
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir = settings.output_dir / job_id

    if demo_pair_id:
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
        pre_path = job_dir / (pre_image.filename or "pre.png")
        post_path = job_dir / (post_image.filename or "post.png")
        with pre_path.open("wb") as f:
            shutil.copyfileobj(pre_image.file, f)
        with post_path.open("wb") as f:
            shutil.copyfileobj(post_image.file, f)
        if pre_path.stat().st_size > MAX_UPLOAD_BYTES or post_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="Image exceeds 25 MB upload limit")

    try:
        mask_path, mode, confidence_path = run_inference(pre_path, post_path, out_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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


@app.post("/brief", response_model=BriefResponse)
async def brief(body: BriefRequest) -> BriefResponse:
    return await generate_brief(body.analysis, body.context)


@app.post("/report/pdf")
async def report_pdf(body: ReportRequest) -> Response:
    pair_id = body.analysis.get("pair_id")
    pdf_bytes = build_field_report_pdf(body.analysis, body.brief, pair_id=pair_id)
    filename = f"disasteriq-report-{pair_id or 'analysis'}.pdf"
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
    brief_resp = await generate_brief(analysis.model_dump(), context)
    return {"analysis": analysis, "brief": brief_resp}
