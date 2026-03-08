import json 
from llama_index.tools.mcp import BasicMCPClient,McpToolSpec
from llama_index.core.agent import FunctionAgent

from src.config.settings import settings 
from src.config.logger import logger 
from src.llm.llm_interface import get_llm 

async def get_job_agent() -> FunctionAgent:

    mcp_client = BasicMCPClient(settings.mcp_server_url)
    mcp_tool_spec = McpToolSpec(client=mcp_client)

    tools = await mcp_tool_spec.to_tool_list_async()

    # --- keep ONLY get_job_details tool ---
    job_tool = None
    for tool in tools:
        if tool.metadata.name == "get_job_details":
            job_tool = tool
            break

    if job_tool is None:
        raise RuntimeError("MCP server did not provide get_job_details tool")

    logger.info("Job agent loaded tool: get_job_details")

    llm = get_llm()

    agent = FunctionAgent(
        tools=[job_tool],   # 🔥 only allow correct tool
        llm=llm,
        verbose=True,
        system_prompt="""
        You are a LinkedIn job posting research assistant.

        Always call the available tool to retrieve the job posting.

        Return ONLY a valid JSON object with this schema:

        {
            "job_title": "exact job title",
            "company": "company name",
            "location": "city or remote",
            "employment_type": "Full-time / Part-time / Contract / etc.",
            "seniority_level": "Entry / Mid / Senior / Director / etc.",
            "description": "full job description text",
            "required_skills": [],
            "preferred_skills": [],
            "required_experience_years": null,
            "required_education": null,
            "responsibilities": [],
            "benefits": [],
            "url": "job URL"
        }

        If a field is not mentioned return null.
        Never invent information.
        """)

    return agent

async def fetch_job_posting(job_url: str) -> dict:

    if "/jobs/view/" not in job_url:
        raise ValueError(
            "Invalid LinkedIn job URL. Use format: "
            "https://www.linkedin.com/jobs/view/{job_id}/"
        )

    logger.info(f"Fetching job posting URL: {job_url}")

    agent = await get_job_agent()

    prompt = f"""
    You MUST call the tool `get_job_details`.

    Arguments:
    url = "{job_url}"

    Return ONLY structured JSON.
    """

    response = await agent.run(prompt)

    raw = str(response).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    raw = raw.strip()

    try:
        data = json.loads(raw)

        logger.info(
            f"Successfully fetched job: "
            f"{data.get('job_title')} @ {data.get('company')}"
        )

        return data

    except json.JSONDecodeError:

        logger.error(f"Job agent returned non-JSON response: {raw[:300]}")

        raise ValueError(
            f"MCP job agent did not return valid JSON for URL: {job_url}. "
            f"Raw response (first 300 chars): {raw[:300]}"
        )