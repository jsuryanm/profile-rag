import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from api.schemas import (
    LoadJobRequest,
    LoadResumeResponse,
    LoadJobResponse,
    AnalyzeRequest,
    StatusResponse,
    AnalyzeResponse,
)

from src.config.logger import logger

from src.services.resume_service import (
    load_resume,
    load_job,
    analyze_resume,
    get_cover_letter,
    get_cert_recommendations,
    get_resume_status,
)

resume_router = APIRouter(prefix="/resume", tags=["Resume Analysis"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



@resume_router.post("/load", response_model=LoadResumeResponse)
async def load_resume_endpoint(
    file: UploadFile = File(...),
    candidate_name: str = Query(default=None),
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_filename

    try:

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"[ResumeRouter] PDF saved to temp: {temp_path}")

        result = await load_resume(str(temp_path), candidate_name)

        return result

    except Exception as e:

        logger.error(f"[ResumeRouter] load_resume failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))

    finally:

        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"[ResumeRouter] Temp PDF cleaned up: {temp_path}")



@resume_router.post("/load-job", response_model=LoadJobResponse)
async def load_job_endpoint(request: LoadJobRequest):

    try:

        result = await load_job(request.job_url)

        return result

    except Exception as e:

        logger.error(f"[ResumeRouter] load_job failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))



@resume_router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(quick: bool = False):

    try:

        result = await analyze_resume(quick=quick)

        return result

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:

        logger.error(f"[ResumeRouter] analyze failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))

@resume_router.get("/cover-letter")
async def cover_letter_endpoint():

    try:

        return await get_cover_letter()

    except Exception as e:

        logger.error(f"[ResumeRouter] get_cover_letter failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))



@resume_router.get("/certifications")
async def certifications_endpoint():

    try:

        return await get_cert_recommendations()

    except Exception as e:

        logger.error(f"[ResumeRouter] get_cert_recommendations failed: {e}")

        raise HTTPException(status_code=500, detail=str(e))



@resume_router.get("/status", response_model=StatusResponse)
def status_endpoint():

    return get_resume_status()