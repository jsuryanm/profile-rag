import json

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool

from src.config.settings import settings
from src.config.logger import logger
from src.llm.llm_interface import get_llm
from src.schemas.agent_outputs import FitAnalysisOutput


# Prompt template for structured_predict — vars are injected at call time
_FIT_ANALYSIS_PROMPT = PromptTemplate(
    """You are a professional technical recruiter.

Analyse how well the candidate's resume matches the job posting using
ONLY the retrieved context below.

Job Posting Context:
{job_context}

Resume Context:
{resume_context}

Important instructions for missing data:
{missing_data_instruction}

Produce a complete fit analysis. fit_score must be an integer 0-100.
If key job requirements are missing, reflect that uncertainty in
score_rationale rather than penalising the candidate."""
)

def build_job_match_agent(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
) -> FunctionAgent:
    """
    Builds the FunctionAgent responsible for retrieving resume and job data.
    The agent now only handles RAG retrieval; structured output is handled
    separately by astructured_predict() in run_job_match_agent().
    """
    llm = get_llm()

    system_prompt = """You are a professional technical recruiter.

    You have two tools:
    - resume_search: retrieves information from the candidate's resume
    - job_search: retrieves information from the job posting

    STRICT RULES:
    1. ALWAYS call job_search first to retrieve required_skills, preferred_skills,
    responsibilities, required_experience_years, and required_education.
    2. ALWAYS call resume_search to retrieve the candidate's skills, experience,
    education, and certifications.
    3. After both tool calls, summarise ALL retrieved information in plain text.
    Do NOT produce JSON — the caller handles output formatting.
    4. Never make up skills or requirements."""

    agent = FunctionAgent(
        tools=[resume_tool, job_tool],
        llm=llm,
        system_prompt=system_prompt,
    )
    logger.info("[JobMatchAgent] Built with resume_search + job_search tools")
    return agent


async def run_job_match_agent(agent: FunctionAgent) -> FitAnalysisOutput:
    llm = get_llm()

    logger.info("[JobMatchAgent] Phase 1: retrieving context via tools")
    raw_summary = str(await agent.run(
        "Call job_search and resume_search, then write a detailed plain-text "
        "summary of all retrieved information."
    )).strip()

    # NEW — inject explicit instructions for handling missing data
    # into the prompt so the LLM doesn't fill gaps with guesses
    result: FitAnalysisOutput = await llm.astructured_predict(
        FitAnalysisOutput,
        _FIT_ANALYSIS_PROMPT,
        job_context=raw_summary,
        resume_context=raw_summary,
        missing_data_instruction=(
            "If required_skills, required_experience_years, or required_education "
            "are not present in the job context, set the corresponding fields to "
            "null or empty list. Do NOT infer or guess missing requirements. "
            "A missing field is not the same as a failed match."
        )
    )

    logger.info(
        f"[JobMatchAgent] fit_score={result.fit_score}/100 | "
        f"matched={len(result.matched_skills)} | "
        f"missing_required={len(result.missing_required_skills)}"
    )
    return result