import json 
from llama_index.tools.mcp import BasicMCPClient,McpToolSpec
from llama_index.core.agent import FunctionAgent

from src.config.settings import settings 
from src.config.logger import logger 
from src.llm.llm_interface import get_llm 

# agent factory

async def get_job_agent() -> FunctionAgent:
    """
    Connects to the LinkedIn MCP server and returns a FunctionAgent
    scoped to job-posting scraping.

    The system prompt instructs the model to return a strict JSON
    schema so downstream processing is deterministic.
    """

    mcp_client = BasicMCPClient(settings.mcp_server_url)
    mcp_tool_spec = McpToolSpec(client=mcp_client)
    tools = await mcp_tool_spec.to_tool_list_async()

    logger.info(f"Job agent loaded {len(tools)} MCP tools: {[t.metadata.name for t in tools]}")

    llm = get_llm()
    agent = FunctionAgent(tools=tools,
                          llm=llm,
                          system_prompt="""You are a LinkedIn job posting research assistant.
        When fetching a job posting you must always:
        1. Use the available tools to fetch the job data from the provided URL.
        2. Deduplicate all content — if something appears more than once, include it once.
        3. Return ONLY a valid JSON object. No markdown, no code blocks, no explanation.
        4. Always use this exact structure:

        {
            "job_title": "exact job title from the posting",
            "company": "company name",
            "location": "city, country or Remote",
            "employment_type": "Full-time / Part-time / Contract / etc.",
            "seniority_level": "Entry / Mid / Senior / Director / etc.",
            "description": "full job description text",
            "required_skills": ["skill1", "skill2"],
            "preferred_skills": ["skill1", "skill2"],
            "required_experience_years": "e.g. 3+ years or null if not stated",
            "required_education": "e.g. Bachelor's in Computer Science or null",
            "responsibilities": ["responsibility1", "responsibility2"],
            "benefits": ["benefit1", "benefit2"],
            "url": "the original job URL"
        }

        If a field is not mentioned in the posting, set it to null.
        Never invent information that is not present in the posting.""",
    )

    return agent

async def fetch_job_posting(job_url: str) -> dict:
    """
    Fetches a LinkedIn job posting and returns it as a structured dict.

    Args:
        job_url: Full LinkedIn job posting URL.

    Returns:
        Parsed job dict matching the schema above.

    Raises:
        ValueError: If the MCP agent returns unparseable content.
    """
    logger.info(f"Fetching job posting URL: {job_url}")
    
    agent = await get_job_agent()
    prompt = (
        f"LinkedIn job posting URL: {job_url}\n"
        "Fetch the full job posting and return it as structured JSON."
    )

    response = await agent.run(prompt)
    raw = str(response).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        logger.info(f"Successfully fetched job: {data.get('job_title')} @ {data.get('company')}")
        return data
    except json.JSONDecodeError:
        logger.error(f"Job agent returned non-JSON response: {raw[:300]}")
        raise ValueError(
            f"MCP job agent did not return valid JSON for URL: {job_url}. "
            f"Raw response (first 300 chars): {raw[:300]}"
        )

