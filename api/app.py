from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.profile_service import load_profile, ask_profile, get_loaded_profile_name
from src.config.logger import logger

app = FastAPI(
    title="Profile RAG API",
    description="Load a LinkedIn profile and ask questions about it.",
    version="1.0.0"
)


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
    use_agent: bool = False  # set True for follow-up conversations

class AskResponse(BaseModel):
    question: str
    answer: str
    profile: str
    mode: str  # "router" or "agentic"


@app.get("/health")
def health():
    loaded = get_loaded_profile_name()
    return {
        "status": "ok",
        "profile_loaded": loaded is not None,
        "loaded_profile": loaded,
    }


@app.post("/profile", response_model=LoadProfileResponse)
async def load_profile_endpoint(request: LoadProfileRequest):
    try:
        result = await load_profile(request.linkedin_url)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """
    Ask a question about the loaded profile.
    
    - use_agent: false (default) — router picks QA vs report tool, no memory
    - use_agent: true — same routing but with conversation memory for follow-ups
    """
    try:
        result = await ask_profile(request.question, use_agent=request.use_agent)
        return {
            "question": request.question,
            "answer": result["answer"],
            "profile": get_loaded_profile_name() or "unknown",
            "mode": result["mode"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))