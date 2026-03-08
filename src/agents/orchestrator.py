import asyncio
from typing import Optional

from pydantic import BaseModel
from llama_index.core.tools import QueryEngineTool

from src.config.logger import logger

from src.agents.job_match_agent import (
    build_job_match_agent,
    run_job_match_agent,
)

from src.agents.recommendations_agent import (
    build_recommendations_agent,
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

from typing import Optional

class AnalysisResult(BaseModel):

    candidate_name: str
    job_title: str
    company: str

    fit_analysis: Optional[FitAnalysisOutput] = None


async def retrieve_shared_context(agent):
    """
    Retrieve resume + job context once so all recommendation tasks
    can reuse the same information.
    """

    query = """
Retrieve the following information.

RESUME
- candidate skills
- technical tools
- work experience
- projects
- education

JOB POSTING
- required skills
- preferred skills
- technologies mentioned
- responsibilities
- qualifications

Return a clear structured summary.
"""

    context = str(await agent.run(query)).strip()

    return context

async def run_full_analysis(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
    candidate_name: str = "the candidate",
    job_title: str = "the role",
    company: str = "the company",
) -> AnalysisResult:
    """
    Executes the full multi-agent pipeline.
    """

    result = AnalysisResult(
        candidate_name=candidate_name,
        job_title=job_title,
        company=company,
    )

    logger.info(
        f"[Orchestrator] Step 1: Fit analysis | "
        f"candidate='{candidate_name}' | job='{job_title}' @ '{company}'"
    )

    match_agent = build_job_match_agent(resume_tool, job_tool)

    result.fit_analysis = await run_job_match_agent(match_agent)

    fit_score = result.fit_analysis.fit_score

    logger.info(
        f"[Orchestrator] Step 1 complete | fit_score={fit_score}/100"
    )


    retrieval_agent = build_recommendations_agent(resume_tool, job_tool)

    logger.info(
        "[Orchestrator] Retrieving shared context once for recommendation tasks"
    )

    shared_context = await retrieve_shared_context(retrieval_agent)

    (
        result.resume_improvements,
        result.cover_letter,
        result.cert_recommendations,
    ) = await asyncio.gather(
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
    )

    logger.info(
        "[Orchestrator] Step 2 complete | All recommendation tasks finished"
    )

    logger.info(
        f"[Orchestrator] Pipeline complete | "
        f"fit_score={fit_score}/100 | "
        f"improvements={len(result.resume_improvements.model_dump())} fields | "
        f"cover_letter_words={result.cover_letter.word_count} | "
        f"certs={len(result.cert_recommendations.certifications)}"
    )

    return result


async def run_fit_score_only(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
    candidate_name: str = "the candidate",
    job_title: str = "the role",
    company: str = "the company",
):
    """
    Runs only the JobMatchAgent (fit score).
    Useful for quick evaluation.
    """

    logger.info(
        f"[Orchestrator] Quick fit score for '{candidate_name}' → '{job_title}'"
    )

    match_agent = build_job_match_agent(resume_tool, job_tool)

    fit_analysis = await run_job_match_agent(match_agent)

    logger.info(
        f"[Orchestrator] Quick fit score: {fit_analysis.fit_score}/100"
    )

    return fit_analysis