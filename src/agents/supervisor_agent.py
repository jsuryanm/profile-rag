import asyncio
from typing import Optional

from pydantic import BaseModel
from llama_index.core.tools import QueryEngineTool

from src.config.logger import logger

from src.agents.retrieval_agent import retrieve_context
from src.agents.job_match_agent import run_job_match_agent
from src.agents.recommendations_agent import (
    run_resume_improvements,
    run_cover_letter,
    run_cert_recommendations,
)

from src.schemas.agent_outputs import (
    FitAnalysisOutput,
    ResumeImprovementsOutput,
    CoverLetterOutput,
    CertRecommendationsOutput,
)


class SupervisorResult(BaseModel):

    candidate_name: str
    job_title: str
    company: str

    fit_analysis: Optional[FitAnalysisOutput] = None
    resume_improvements: Optional[ResumeImprovementsOutput] = None
    cover_letter: Optional[CoverLetterOutput] = None
    cert_recommendations: Optional[CertRecommendationsOutput] = None


async def run_supervisor_pipeline(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
    candidate_name: str,
    job_title: str,
    company: str,
):

    logger.info("[SupervisorAgent] Starting analysis pipeline")

    result = SupervisorResult(
        candidate_name=candidate_name,
        job_title=job_title,
        company=company,
    )

    # -----------------------------------------
    # Step 1 — Retrieval Agent
    # -----------------------------------------

    resume_context, job_context = await retrieve_context(
        resume_tool,
        job_tool,
    )

    # -----------------------------------------
    # Step 2 — Fit Analysis Agent
    # -----------------------------------------

    logger.info("[SupervisorAgent] Running fit analysis")

    result.fit_analysis = await run_job_match_agent(
        resume_context,
        job_context,
    )

    fit_score = result.fit_analysis.fit_score

    logger.info(
        f"[SupervisorAgent] Fit score = {fit_score}"
    )

    shared_context = f"""
RESUME CONTEXT
{resume_context}

JOB CONTEXT
{job_context}
"""

    # -----------------------------------------
    # Step 3 — Recommendation Agents (parallel)
    # -----------------------------------------

    logger.info("[SupervisorAgent] Running recommendation agents")

    tasks = await asyncio.gather(

        run_resume_improvements(
            shared_context,
            result.fit_analysis,
            candidate_name=candidate_name,
        ),

        run_cover_letter(
            shared_context,
            result.fit_analysis,
            candidate_name=candidate_name,
            job_title=job_title,
            company=company,
        ),

        run_cert_recommendations(
            shared_context,
            result.fit_analysis,
        ),

        return_exceptions=True   # IMPORTANT FIX
    )

    # -----------------------------------------
    # Handle failures safely
    # -----------------------------------------

    if isinstance(tasks[0], Exception):
        logger.warning(
            f"[SupervisorAgent] Resume improvements failed: {tasks[0]}"
        )
        result.resume_improvements = None
    else:
        result.resume_improvements = tasks[0]

    if isinstance(tasks[1], Exception):
        logger.warning(
            f"[SupervisorAgent] Cover letter generation failed: {tasks[1]}"
        )
        result.cover_letter = None
    else:
        result.cover_letter = tasks[1]

    if isinstance(tasks[2], Exception):
        logger.warning(
            f"[SupervisorAgent] Certification recommendations failed: {tasks[2]}"
        )
        result.cert_recommendations = None
    else:
        result.cert_recommendations = tasks[2]

    logger.info("[SupervisorAgent] Pipeline complete")

    return result