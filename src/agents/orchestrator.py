from llama_index.core.tools import QueryEngineTool

from src.config.logger import logger
from src.agents.job_match_agent import run_job_match_agent
from src.agents.retrieval_agent import retrieve_context


async def run_fit_score_only(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
    candidate_name: str,
    job_title: str,
    company: str,
):

    logger.info(
        f"[Orchestrator] Quick fit score for '{candidate_name}' → '{job_title}'"
    )

    # reuse RetrievalAgent
    resume_context, job_context = await retrieve_context(
        resume_tool,
        job_tool,
    )

    fit_analysis = await run_job_match_agent(
        resume_context,
        job_context,
    )

    logger.info(
        f"[Orchestrator] Quick fit score: {fit_analysis.fit_score}/100"
    )

    return fit_analysis