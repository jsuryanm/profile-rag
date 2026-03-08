from pydantic import BaseModel 
from typing import List,Optional 

class ExperienceAssessment(BaseModel):
    required_years: Optional[str]
    candidate_estimated_years: Optional[str]
    gap: Optional[str]


class EducationMatch(BaseModel):
    required: Optional[str]
    candidate_has: Optional[str]
    matches: Optional[bool]


class FitAnalysis(BaseModel):
    fit_score: int
    score_rationale: str

    matched_skills: List[str]
    missing_required_skills: List[str]
    missing_preferred_skills: List[str]

    experience_assessment: ExperienceAssessment
    education_match: EducationMatch

    strengths: List[str]
    weaknesses: List[str]