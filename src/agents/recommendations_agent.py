from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool

from src.config.logger import logger
from src.llm.llm_interface import get_llm

from src.schemas.agent_outputs import (
    ResumeImprovementsOutput,
    CoverLetterOutput,
    CertRecommendationsOutput,
    FitAnalysisOutput,
)


_IMPROVEMENTS_PROMPT = PromptTemplate(
"""
You are a senior resume consultant.

Use the context below to suggest improvements to the candidate's resume.

Context:
{shared_context}

Candidate gaps:
- Missing required skills: {missing_required}
- Missing preferred skills: {missing_preferred}
- Weaknesses: {weaknesses}

Current fit score: {fit_score}/100

Return specific actionable improvements.
Avoid generic advice.
"""
)

_COVER_LETTER_PROMPT = PromptTemplate(
"""
You are a professional career coach writing a cover letter.

Candidate: {candidate_name}
Role: {job_title} at {company}

Fit score: {fit_score}/100

Context:
{shared_context}

Strengths:
{strengths}

Matched skills:
{matched_skills}

Write a strong 3-paragraph cover letter.
Do not mention the fit score.
"""
)

_CERT_PROMPT = PromptTemplate(
"""
You are a career advisor recommending certifications.

Context:
{shared_context}

Skill gaps:
- Missing required skills: {missing_required}
- Missing preferred skills: {missing_preferred}
- Experience gap: {experience_gap}

Recommend relevant certifications or online courses.
Prioritize widely recognized credentials.
"""
)


def build_recommendations_agent(
    resume_tool: QueryEngineTool,
    job_tool: QueryEngineTool,
) -> FunctionAgent:

    llm = get_llm()

    system_prompt = """
You are a career analysis assistant.

You have two tools:
- resume_search
- job_search

Always retrieve information from both tools before answering.

Return a clear text summary of:
- candidate skills
- experience
- education
- job requirements
- technologies
"""

    agent = FunctionAgent(
        tools=[resume_tool, job_tool],
        llm=llm,
        system_prompt=system_prompt,
    )

    logger.info("[RecommendationsAgent] Built with resume_search + job_search")

    return agent



async def run_resume_improvements(
    shared_context: str,
    fit_analysis: FitAnalysisOutput,
    candidate_name: str,
) -> ResumeImprovementsOutput:

    llm = get_llm()

    logger.info("[RecommendationsAgent] Generating resume improvements")

    result: ResumeImprovementsOutput = await llm.astructured_predict(
        ResumeImprovementsOutput,
        _IMPROVEMENTS_PROMPT,
        shared_context=shared_context,
        missing_required=fit_analysis.missing_required_skills,
        missing_preferred=fit_analysis.missing_preferred_skills,
        weaknesses=fit_analysis.weaknesses,
        fit_score=fit_analysis.fit_score,
    )

    return result



async def run_cover_letter(
    shared_context: str,
    fit_analysis: FitAnalysisOutput,
    candidate_name: str,
    job_title: str,
    company: str,
) -> CoverLetterOutput:

    llm = get_llm()

    logger.info("[RecommendationsAgent] Generating cover letter")

    result: CoverLetterOutput = await llm.astructured_predict(
        CoverLetterOutput,
        _COVER_LETTER_PROMPT,
        candidate_name=candidate_name,
        job_title=job_title,
        company=company,
        fit_score=fit_analysis.fit_score,
        shared_context=shared_context,
        strengths=fit_analysis.strengths,
        matched_skills=fit_analysis.matched_skills,
    )

    return result



async def run_cert_recommendations(
    shared_context: str,
    fit_analysis: FitAnalysisOutput,
) -> CertRecommendationsOutput:

    llm = get_llm()

    logger.info("[RecommendationsAgent] Generating certification recommendations")

    result: CertRecommendationsOutput = await llm.astructured_predict(
        CertRecommendationsOutput,
        _CERT_PROMPT,
        shared_context=shared_context,
        missing_required=fit_analysis.missing_required_skills,
        missing_preferred=fit_analysis.missing_preferred_skills,
        experience_gap=fit_analysis.experience_assessment.gap or "unknown",
    )

    return result