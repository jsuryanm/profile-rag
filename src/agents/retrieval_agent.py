import asyncio
from llama_index.core.tools import QueryEngineTool
from src.config.logger import logger


async def retrieve_context(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
):
    """
    Retrieval Agent

    Retrieves resume + job context in parallel.
    """

    resume_query = """
Summarize the candidate's:
- skills
- tools
- work experience
- projects
- education
"""

    job_query = """
Summarize the job posting including:
- required skills
- preferred skills
- responsibilities
- technologies mentioned
- qualifications
"""

    logger.info("[RetrievalAgent] Retrieving resume + job context")

    resume_context, job_context = await asyncio.gather(
        resume_tool.query_engine.aquery(resume_query),
        job_tool.query_engine.aquery(job_query),
    )

    return str(resume_context), str(job_context)