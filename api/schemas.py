from pydantic import BaseModel
from src.agents.orchestrator import AnalysisResult

class LoadProfileRequest(BaseModel):
    linkedin_url: str

class LoadProfileResponse(BaseModel):
    status: str
    name: str
    headline: str
    location: str
    chunks_indexed: int

class AskRequest(BaseModel):
    question: str
    use_agent: bool = False

class AskResponse(BaseModel):
    question: str
    answer: str
    profile: str
    mode: str

class AnalyzeRequest(BaseModel):
    quick: bool = False

class AnalyzeResponse(BaseModel):
    mode: str
    analysis: AnalysisResult


class LoadJobRequest(BaseModel):
    job_url: str  # LinkedIn job posting URL


class LoadResumeResponse(BaseModel):
    status: str
    candidate_name: str
    chunks_indexed: int


class LoadJobResponse(BaseModel):
    status: str
    job_title: str
    company: str
    location: str
    chunks_indexed: int


class AnalyzeRequest(BaseModel):
    quick: bool = False  # True = fit score only, False = full pipeline


class StatusResponse(BaseModel):
    resume_loaded: bool
    job_loaded: bool
    candidate_name: str | None
    job_title: str | None
    company: str | None
    has_cached_result: bool
