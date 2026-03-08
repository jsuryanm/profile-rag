from fastapi import FastAPI, HTTPException
from api.schemas import (LoadProfileResponse,
                         LoadProfileRequest,
                         AskResponse,
                         AskRequest)

from api.resume_router import resume_router

from src.services.profile_service import load_profile, ask_profile, get_loaded_profile_name
from src.config.logger import logger


app = FastAPI(
    title="Profile RAG API",
    description="Load a LinkedIn profile and ask questions about it. Analyse resumes against job postings.",
    version="2.0.0"
)

# NEW: mount the resume router (all endpoints prefixed with /resume)
app.include_router(resume_router)


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
    try:
        result = await ask_profile(request.question, use_agent=request.use_agent)

        # result["answer"] is now a ProfileAnswerOutput Pydantic model
        # so we call .model_dump() to get the plain dict, then extract the answer
        answer_data = result["answer"]
        answer_text = (
            answer_data.answer               # attribute access if Pydantic model
            if hasattr(answer_data, "answer")
            else str(answer_data)            # fallback for plain string
        )

        return {
            "question": request.question,
            "answer": answer_text,
            "profile": get_loaded_profile_name() or "unknown",
            "mode": result["mode"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))