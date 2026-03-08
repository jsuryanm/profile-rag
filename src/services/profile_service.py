from src.config.logger import logger
from mcp_client.linkedin_client import fetch_profile_agent
from src.processing.data_processing import process_profile
from src.rag.query_engine import (
    build_router_query_engine,
    build_agentic_rag,
    query_profile,
    query_profile_agentic,
)
from src.rag.eval import evaluate_router
from src.schemas.agent_outputs import LinkedInProfileOutput

from llama_index.core.memory import ChatMemoryBuffer
import asyncio

_state = {
    "router": None,
    "agent": None,
    "subject_name": None,
    "memory": None,
}


def log_eval_result(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.warning("[Eval] Background evaluation task was cancelled")
    elif task.exception():
        logger.error(f"[Eval] Background evaluation task raised: {task.exception()}")
    else:
        logger.info("[Eval] Background evaluation task completed successfully")


async def load_profile(linkedin_url: str) -> dict:
    profile: LinkedInProfileOutput = await fetch_profile_agent(
        user_query="Fetch the full LinkedIn profile including experience and education",
        linkedin_url=linkedin_url,
    )

    subject_name = profile.name
    nodes = process_profile(
        profile.model_dump(),
        metadata={"source": "linkedin", "url": linkedin_url},
    )

    router = build_router_query_engine(nodes, subject_name=subject_name)

    if _state["subject_name"] != subject_name:
        _state["memory"] = ChatMemoryBuffer.from_defaults(token_limit=4096)

    agent = build_agentic_rag(router, subject_name=subject_name, memory=_state["memory"])

    _state["router"] = router
    _state["agent"] = agent
    _state["subject_name"] = subject_name

    eval_task = asyncio.create_task(evaluate_router(router, subject_name=subject_name))
    eval_task.add_done_callback(log_eval_result)

    # NEW — include missing_sections so the API response tells the caller
    # exactly what data was unavailable, rather than silently returning nulls
    response = {
        "status": "loaded",
        "name": subject_name,
        "headline": profile.headline or "",
        "location": profile.location or "",
        "chunks_indexed": len(nodes),
    }

    if profile.missing_sections:
        response["warning"] = (
            f"Some profile sections were unavailable: "
            f"{', '.join(profile.missing_sections)}. "
            f"Answers about these sections may be incomplete."
        )
        logger.warning(
            f"[ProfileService] Profile incomplete for {subject_name}: "
            f"missing {profile.missing_sections}"
        )

    return response


async def ask_profile(question: str, use_agent: bool = False) -> dict:
    """
    Query the loaded profile.
    - use_agent=False: stateless router query, no memory
    - use_agent=True: agentic query with conversation memory
    """
    if _state["router"] is None:
        raise ValueError("No profile loaded. Call POST /profile first.")

    if use_agent:
        answer = await query_profile_agentic(_state["agent"], question)
        mode = "agentic"
    else:
        answer = await query_profile(
            _state["router"],
            question,
            subject_name=_state["subject_name"],
        )
        mode = "router"

    return {"answer": answer, "mode": mode}


def get_loaded_profile_name() -> str | None:
    return _state["subject_name"]