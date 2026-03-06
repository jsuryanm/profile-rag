from src.config.logger import logger
from mcp_client.linkedin_client import fetch_profile_agent
from src.processing.data_processing import process_profile
from src.rag.query_engine import (build_router_query_engine,
                                  build_agentic_rag,
                                  query_profile,
                                  query_profile_agentic)

from llama_index.core.memory import ChatMemoryBuffer

_state = {
    "router": None,
    "agent": None,
    "subject_name": None,
    "memory":None
}


async def load_profile(linkedin_url: str) -> dict:
    """Fetch, chunk, index, and build router + agent."""
    logger.info(f"Loading profile for URL: {linkedin_url}")

    profile_data = await fetch_profile_agent(
        user_query="Fetch the full LinkedIn profile including experience and education",
        linkedin_url=linkedin_url
    )

    if not profile_data or "raw" in profile_data:
        raise ValueError(f"Failed to fetch a valid profile from: {linkedin_url}")

    subject_name = profile_data.get("name", "Unknown")
    logger.info(f"Successfully fetched profile for: {subject_name}")

    nodes = process_profile(
        profile_data,
        metadata={"source": "linkedin", "url": linkedin_url}
    )

    router = build_router_query_engine(nodes, subject_name=subject_name)

    if _state["subject_name"] != subject_name:
        _state["memory"] = ChatMemoryBuffer.from_defaults(token_limit=4096)
        logger.info("New profile detected, conversation memory cleared")

    agent = build_agentic_rag(router, subject_name=subject_name,memory=_state["memory"])

    _state["router"] = router
    _state["agent"] = agent
    _state["subject_name"] = subject_name

    return {
        "status": "loaded",
        "name": subject_name,
        "headline": profile_data.get("headline", ""),
        "location": profile_data.get("location", ""),
        "chunks_indexed": len(nodes),
    }


async def ask_profile(question: str, use_agent: bool = False) -> dict:
    """
    Query the loaded profile.
    - use_agent=False (default): stateless router query, no memory
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
            subject_name=_state["subject_name"]
        )
        mode = "router"

    return {"answer": answer, "mode": mode}


def get_loaded_profile_name() -> str | None:
    return _state["subject_name"]