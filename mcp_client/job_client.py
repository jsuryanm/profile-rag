import json 
from llama_index.tools.mcp import BasicMCPClient,McpToolSpec
from llama_index.core.agent import FunctionAgent
from src.schemas.agent_outputs import JobPostingOutput
from llama_index.core.prompts import PromptTemplate
from src.config.settings import settings 
from src.config.logger import logger 
from src.llm.llm_interface import get_llm 

_JOB_PROMPT = PromptTemplate(
"""
You are a LinkedIn job posting parser.

Using ONLY the raw job posting text below, extract structured data.

If a field is missing return null or an empty list.

Raw Job Posting:
{raw_job}
"""
)

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

        IMPORTANT: Return ONLY a raw JSON object. 
        - No markdown formatting
        - No code fences (no ```)  
        - No explanation text before or after
        - Start your response with { and end with }


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
    llm = get_llm()

    prompt = f"""
    You MUST call the tool `get_job_details`.

    Arguments:
    url = "{job_url}"

    Return the FULL job posting text.
    Do NOT format it as JSON.
    """

    # Step 1 — retrieve raw job posting
    raw = str(await agent.run(prompt)).strip()

    logger.info("[JobClient] Phase 1 complete — raw job text retrieved")

    # Step 2 — structured output parsing
    job: JobPostingOutput = await llm.astructured_predict(
        JobPostingOutput,
        _JOB_PROMPT,
        raw_job=raw,
    )

    logger.info(
        f"[JobClient] Parsed job: {job.job_title} @ {job.company}"
    )

    return job.model_dump()