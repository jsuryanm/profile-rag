from llama_index.core.prompts import PromptTemplate

from src.llm.llm_interface import get_llm
from src.schemas.agent_outputs import FitAnalysisOutput
from src.config.logger import logger


FIT_ANALYSIS_PROMPT = PromptTemplate(
"""
You are an expert technical recruiter.

Analyze how well a candidate's resume matches the job posting.

RESUME CONTEXT
{resume_context}

JOB POSTING CONTEXT
{job_context}

Evaluate the candidate against the job requirements.

Return a structured analysis with:

- fit_score (0–100)
- score_rationale
- matched_skills
- missing_required_skills
- missing_preferred_skills
- strengths
- weaknesses

Rules:
- Base the score strictly on alignment with required skills and experience.
- Required skills missing should significantly reduce the score.
- Preferred skills missing should slightly reduce the score.
- If a skill is mentioned indirectly, count it as matched.
"""
)


async def run_job_match_agent(
    resume_context: str,
    job_context: str,
) -> FitAnalysisOutput:
    """
    Performs job matching analysis using a single structured LLM call.

    This replaces the slower multi-step agent loop and significantly
    reduces latency while still producing a structured result.

    Args:
        resume_context: Retrieved resume information
        job_context: Retrieved job posting information

    Returns:
        FitAnalysisOutput
    """

    llm = get_llm()

    logger.info("[JobMatchAgent] Running structured fit analysis")

    try:

        result: FitAnalysisOutput = await llm.astructured_predict(
            FitAnalysisOutput,
            FIT_ANALYSIS_PROMPT,
            resume_context=resume_context,
            job_context=job_context,
        )

        logger.info(
            f"[JobMatchAgent] fit_score={result.fit_score}/100 | "
            f"matched={len(result.matched_skills)} | "
            f"missing_required={len(result.missing_required_skills)}"
        )

        return result

    except Exception as e:

        logger.error(f"[JobMatchAgent] Fit analysis failed: {e}")

        raise RuntimeError(
            "Failed to generate job match analysis."
        )