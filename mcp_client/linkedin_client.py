import json
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.prompts import PromptTemplate
from src.config.logger import logger
from src.llm.llm_interface import get_llm
from src.schemas.agent_outputs import LinkedInProfileOutput
from src.config.settings import settings


# Prompt template for structured_predict
_PROFILE_PROMPT = PromptTemplate(
    """You are a LinkedIn profile research assistant.

Using ONLY the raw profile data below, extract and structure
all available information. If a field is not present, use null.

Raw profile data:
{raw_profile}
"""
)


async def get_linkedin_agent() -> FunctionAgent:
    mcp_client = BasicMCPClient(settings.mcp_server_url)
    mcp_tool_spec = McpToolSpec(client=mcp_client)
    tools = await mcp_tool_spec.to_tool_list_async()

    logger.info(
        f"Loaded {len(tools)} tools from Gradio MCP: "
        f"{[t.metadata.name for t in tools]}"
    )

    llm = get_llm()

    # Agent only handles retrieval — structured output is handled
    # separately by astructured_predict() in fetch_profile_agent()
    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt="""You are a LinkedIn profile research assistant.
        Use the available tools to fetch the profile from the given URL.
        Return the raw fetched content as plain text — do not format or
        summarise it. The caller will handle structuring.""",
    )
    return agent


async def fetch_profile_agent(
    user_query: str,
    linkedin_url: str,
) -> LinkedInProfileOutput:
    """
    Phase 1 — Agent fetches raw profile data via MCP tool.
    Phase 2 — astructured_predict() parses it into LinkedInProfileOutput.

    Returns:
        LinkedInProfileOutput — validated Pydantic model.
    """
    llm = get_llm()
    agent = await get_linkedin_agent()

    # Phase 1: raw retrieval
    prompt = f"LinkedIn profile URL: {linkedin_url}\nUser request: {user_query}"
    logger.info(f"[LinkedInClient] Phase 1: fetching profile for {linkedin_url}")
    raw = str(await agent.run(prompt)).strip()

    # Phase 2: structured output
    logger.info("[LinkedInClient] Phase 2: structured_predict → LinkedInProfileOutput")
    result: LinkedInProfileOutput = await llm.astructured_predict(
        LinkedInProfileOutput,
        _PROFILE_PROMPT,
        raw_profile=raw,
    )

    logger.info(f"[LinkedInClient] Profile parsed for: {result.name}")
    return result