from pydantic import BaseModel
from typing import List


class SkillSuggestion(BaseModel):
    skill: str
    how_to_demonstrate: str


class ResumeImprovements(BaseModel):
    summary_improvements: List[str]
    skills_to_add: List[SkillSuggestion]
    experience_reframing: List[str]
    keywords_to_include: List[str]
    sections_to_add: List[str]
    overall_priority: str


class CoverLetter(BaseModel):
    cover_letter: str
    key_talking_points: List[str]
    tone: str
    word_count: int


class Certification(BaseModel):
    name: str
    provider: str
    addresses_skill: str
    estimated_duration: str
    priority: str
    url_hint: str | None


class OnlineCourse(BaseModel):
    name: str
    platform: str
    addresses_skill: str
    estimated_duration: str
    priority: str


class CertificationRecommendations(BaseModel):
    certifications: List[Certification]
    online_courses: List[OnlineCourse]
    learning_path_summary: str