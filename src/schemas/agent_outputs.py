from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# JobMatchAgent 

class ExperienceAssessmentOutput(BaseModel):
    required_years: Optional[str] = Field(
        default=None,
        description="Years of experience required by the job posting, or null."
    )
    candidate_estimated_years: Optional[str] = Field(
        default=None,
        description="Estimated years of experience the candidate has."
    )
    gap: Optional[str] = Field(
        default=None,
        description="'+2 years' means candidate exceeds; '-1 year' means deficit."
    )


class EducationMatchOutput(BaseModel):
    required: Optional[str] = Field(default=None)
    candidate_has: Optional[str] = Field(default=None)
    matches: Optional[bool] = Field(default=None)


class FitAnalysisOutput(BaseModel):
    """Structured output for JobMatchAgent — replaces manual json.loads()."""
    fit_score: int = Field(description="Overall fit score 0-100.")
    score_rationale: str = Field(description="2-3 sentence explanation of the score.")
    matched_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)
    experience_assessment: ExperienceAssessmentOutput = Field(
        default_factory=ExperienceAssessmentOutput
    )
    education_match: EducationMatchOutput = Field(
        default_factory=EducationMatchOutput
    )
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


# RecommendationsAgent: resume improvements 

class SkillToAddOutput(BaseModel):
    skill: str
    how_to_demonstrate: str = Field(
        description="Concrete suggestion, e.g. 'add a side project using X'."
    )


class ResumeImprovementsOutput(BaseModel):
    """Structured output for run_resume_improvements()."""
    summary_improvements: List[str] = Field(default_factory=list)
    skills_to_add: List[SkillToAddOutput] = Field(default_factory=list)
    experience_reframing: List[str] = Field(default_factory=list)
    keywords_to_include: List[str] = Field(default_factory=list)
    sections_to_add: List[str] = Field(default_factory=list)
    overall_priority: str = Field(
        description="The single most impactful change to make first."
    )


# RecommendationsAgent — cover letter

class CoverLetterOutput(BaseModel):
    """Structured output for run_cover_letter()."""
    cover_letter: str = Field(
        description="Full 3-paragraph cover letter. Use \\n for line breaks."
    )
    key_talking_points: List[str] = Field(default_factory=list)
    tone: str = Field(description="professional | conversational | technical")
    word_count: int


# RecommendationsAgent — certifications

class CertificationOutput(BaseModel):
    name: str
    provider: str
    addresses_skill: str
    estimated_duration: str
    priority: str = Field(description="high | medium | low")
    url_hint: Optional[str] = Field(default=None)


class OnlineCourseOutput(BaseModel):
    name: str
    platform: str
    addresses_skill: str
    estimated_duration: str
    priority: str = Field(description="high | medium | low")


class CertRecommendationsOutput(BaseModel):
    """Structured output for run_cert_recommendations()."""
    certifications: List[CertificationOutput] = Field(default_factory=list)
    online_courses: List[OnlineCourseOutput] = Field(default_factory=list)
    learning_path_summary: str

# LinkedIn Profile Agent output 

class CurrentRoleOutput(BaseModel):
    title: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    duration: Optional[str] = Field(default=None)


class ExperienceItemOutput(BaseModel):
    title: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    duration: Optional[str] = Field(default=None)


class EducationItemOutput(BaseModel):
    school: Optional[str] = Field(default=None)
    degree: Optional[str] = Field(default=None)
    years: Optional[str] = Field(default=None)


class LinkedInProfileOutput(BaseModel):
    name: str = Field(description="Full name. Required.")
    headline: Optional[str] = Field(
        default=None,
        description="Current job title and company. Set null if not found."
    )
    location: Optional[str] = Field(
        default=None,
        description="City and country. Set null if not found."
    )
    current_role: Optional[CurrentRoleOutput] = Field(
        default=None,
        description="Most recent role. Set null if experience section is empty."
    )
    experience: List[ExperienceItemOutput] = Field(
        default_factory=list,
        description="All work experience entries. Empty list if none found."
    )
    education: List[EducationItemOutput] = Field(
        default_factory=list,
        description="All education entries. Empty list if none found."
    )

    missing_sections: List[str] = Field(
        default_factory=list,
        description=(
            "List the names of any sections that were absent or empty in the "
            "raw data, e.g. ['experience', 'education']. Empty list if complete."
        )
    )

class ProfileFactsOutput(BaseModel):
    """
    Structured output for the initial facts generation.
    Replaces the raw text string currently returned.
    """
    facts: List[str] = Field(
        description="3 interesting facts about this person's career or education.",
        min_length=3,
        max_length=3,
    )


class ProfileReportOutput(BaseModel):
    """
    Structured output for the full profile report.
    """
    career_summary: str = Field(
        description="2-3 sentence overview of the person's career trajectory."
    )
    icebreaker_questions: List[str] = Field(
        description="3 specific, thoughtful questions to start a meaningful conversation.",
        min_length=3,
        max_length=3,
    )
    networking_tips: List[str] = Field(
        description="Specific suggestions for how best to approach and connect with them.",
    )


class ProfileAnswerOutput(BaseModel):
    """
    Structured output for direct factual Q&A about the profile.
    """
    answer: str = Field(
        description="Direct answer to the question using only the profile context."
    )
    confidence: str = Field(
        description="high | medium | low — how confident based on available context."
    )
    source_hint: Optional[str] = Field(
        default=None,
        description="Which part of the profile this came from, e.g. 'experience section'."
    )

class JobPostingOutput(BaseModel):
    job_title: str = Field(description="Exact job title. Required.")
    company: str = Field(description="Company name. Required.")
    location: Optional[str] = Field(default=None)
    employment_type: Optional[str] = Field(default=None)
    seniority_level: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    required_skills: List[str] = Field(
        default_factory=list,
        description="Required skills. Empty list if not explicitly stated."
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Preferred skills. Empty list if not stated."
    )
    required_experience_years: Optional[str] = Field(
        default=None,
        description="e.g. '3+ years'. Null if not stated — do not guess."
    )
    required_education: Optional[str] = Field(
        default=None,
        description="e.g. 'Bachelor in CS'. Null if not stated — do not guess."
    )
    responsibilities: List[str] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)
    url: Optional[str] = Field(default=None)
    # NEW
    missing_sections: List[str] = Field(
        default_factory=list,
        description=(
            "List any fields that were absent from the posting, "
            "e.g. ['required_experience_years', 'benefits']. "
            "Empty list if the posting was complete."
        )
    )
