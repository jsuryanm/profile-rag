from pathlib import Path

from src.config.logger import logger
from src.processing.resume_processing import process_resume

from src.rag.resume_index import (
    build_resume_tool,
    build_job_tool,
    process_job_posting,
)

from src.agents.supervisor_agent import run_supervisor_pipeline
from src.agents.orchestrator import run_fit_score_only

from src.schemas.agent_outputs import FitAnalysisOutput

from mcp_client.job_client import fetch_job_posting


_state: dict = {
    "resume_tool": None,
    "job_tool": None,
    "candidate_name": None,
    "job_title": None,
    "company": None,
    "last_result": None,
}


# ---------------------------------------------------
# LOAD RESUME
# ---------------------------------------------------
async def load_resume(pdf_path: str, candidate_name: str = None) -> dict:

    pdf_path = Path(pdf_path)

    if not candidate_name:
        candidate_name = pdf_path.stem.replace("_", " ").replace("-", " ").title()

    logger.info(f"[ResumeService] Loading resume for: {candidate_name}")

    nodes = process_resume(pdf_path, candidate_name=candidate_name)

    resume_tool = build_resume_tool(nodes, candidate_name=candidate_name)

    _state["resume_tool"] = resume_tool
    _state["candidate_name"] = candidate_name
    _state["last_result"] = None

    logger.info(
        f"[ResumeService] Resume indexed | "
        f"candidate='{candidate_name}' | chunks={len(nodes)}"
    )

    return {
        "status": "resume_loaded",
        "candidate_name": candidate_name,
        "chunks_indexed": len(nodes),
    }


# ---------------------------------------------------
# LOAD JOB
# ---------------------------------------------------
async def load_job(job_url: str) -> dict:

    logger.info(f"[ResumeService] Fetching job posting: {job_url}")

    job_data = await fetch_job_posting(job_url)

    job_title = job_data.get("job_title", "Unknown Role")
    company = job_data.get("company", "Unknown Company")

    logger.info(f"[ResumeService] Job fetched: {job_title} @ {company}")

    nodes = process_job_posting(job_data)

    job_tool = build_job_tool(nodes, job_title=job_title)

    _state["job_tool"] = job_tool
    _state["job_title"] = job_title
    _state["company"] = company
    _state["last_result"] = None

    return {
        "status": "job_loaded",
        "job_title": job_title,
        "company": company,
        "location": job_data.get("location", ""),
        "chunks_indexed": len(nodes),
    }


# ---------------------------------------------------
# ANALYZE RESUME
# ---------------------------------------------------
async def analyze_resume(quick: bool = False) -> dict:

    _assert_both_loaded()

    candidate_name = _state["candidate_name"]
    job_title = _state["job_title"]
    company = _state["company"]

    resume_tool = _state["resume_tool"]
    job_tool = _state["job_tool"]

    # --------------------------------
    # QUICK MODE
    # --------------------------------
    if quick:

        logger.info("[ResumeService] Running quick fit score only")

        fit_analysis: FitAnalysisOutput = await run_fit_score_only(
            resume_tool,
            job_tool,
            candidate_name=candidate_name,
            job_title=job_title,
            company=company,
        )

        _state["last_result"] = fit_analysis

        return {
            "mode": "quick",
            "analysis": {
                "candidate_name": candidate_name,
                "job_title": job_title,
                "company": company,
                "fit_analysis": fit_analysis.model_dump(),
            },
        }

    # --------------------------------
    # FULL PIPELINE (SupervisorAgent)
    # --------------------------------

    logger.info("[ResumeService] Running supervisor agent pipeline")

    result = await run_supervisor_pipeline(
        resume_tool,
        job_tool,
        candidate_name=candidate_name,
        job_title=job_title,
        company=company,
    )

    _state["last_result"] = result

    return {
        "mode": "full",
        "analysis": result.model_dump(),
    }


# ---------------------------------------------------
# COVER LETTER
# ---------------------------------------------------
async def get_cover_letter() -> dict:

    if _state["last_result"] is None:
        logger.info("[ResumeService] No cached result — running full analysis")
        await analyze_resume()

    result = _state["last_result"]

    return {
        "candidate_name": _state["candidate_name"],
        "job_title": _state["job_title"],
        "company": _state["company"],
        "cover_letter": result.cover_letter.model_dump()
        if result.cover_letter
        else {},
    }


# ---------------------------------------------------
# CERT RECOMMENDATIONS
# ---------------------------------------------------
async def get_cert_recommendations() -> dict:

    if _state["last_result"] is None:
        logger.info("[ResumeService] No cached result — running full analysis")
        await analyze_resume()

    result = _state["last_result"]

    return {
        "candidate_name": _state["candidate_name"],
        "job_title": _state["job_title"],
        "cert_recommendations": result.cert_recommendations.model_dump()
        if result.cert_recommendations
        else {},
    }


# ---------------------------------------------------
# STATUS
# ---------------------------------------------------
def get_resume_status() -> dict:

    return {
        "resume_loaded": _state["resume_tool"] is not None,
        "job_loaded": _state["job_tool"] is not None,
        "candidate_name": _state["candidate_name"],
        "job_title": _state["job_title"],
        "company": _state["company"],
        "has_cached_result": _state["last_result"] is not None,
    }


# ---------------------------------------------------
# INTERNAL CHECK
# ---------------------------------------------------
def _assert_both_loaded():

    if _state["resume_tool"] is None:
        raise ValueError("No resume loaded. Call POST /resume/load first.")

    if _state["job_tool"] is None:
        raise ValueError("No job posting loaded. Call POST /resume/load-job first.")