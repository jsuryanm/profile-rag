import asyncio 
from src.config.logger import logger 
from src.config.settings import settings 

from llama_index.tools.mcp import BasicMCPClient,McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from src.llm.llm_interface import get_llm

# FunctionAgent is a tool-using agent (it decides when and which tool call,args to send)
# McpToolspec converts mcp tools to llamaindex tools

async def get_linkedin_agent() -> FunctionAgent:
    """
    Connects to the LinkedIn MCP server and returns a LlamaIndex
    FunctionAgent that can call LinkedIn tools based on user intent.
    """
    mcp_client = BasicMCPClient(settings.mcp_server_url)
    mcp_tool_spec = McpToolSpec(client=mcp_client)
    tools = await mcp_tool_spec.to_tool_list_async()

    logger.info(
        f"Loaded {len(tools)} tools from LinkedIn MCP server: "
        f"{[t.metadata.name for t in tools]}"
    )
    llm = get_llm()
    agent = FunctionAgent(tools=tools,
                          llm=llm,
                          system_prompt="""You are a LinkedIn profile research assistant. 
                          When fetching and returning the profile data, you must always:
                          1. Use the available tools to fetch requested profile data.
                          2. Deduplicate all content - if a piece of information appears more than once, include it only once.
                          3. Return ONLY a valid JSON object with no markdown, no code blocks, no extra explanation.
                          4. Always structure the response in this format: 
                          
                          {
                            "name": "full name",
                            "headline": "current job title and company",
                            "location": "city, country",
                            "current_role": {
                                "title": "job title",
                                "company": "company name",
                                "duration": "start date to present"
                            },
                            "experience": [
                                {
                                    "title": "job title",
                                    "company": "company name",
                                    "duration": "date range"
                                }
                            ],
                            "education": [
                                {
                                    "school": "school name",
                                    "degree": "degree and field",
                                    "years": "year range"
                                }
                            ]
                        }"""
                        )
    
    return agent

async def fetch_profile_agent(user_query: str, linkedin_url: str) -> str:
    """
    Main entry point: given a user query and LinkedIn URL,
    the agent decides which MCP tool(s) to call and returns the raw result.
    """
    agent = await get_linkedin_agent()
    prompt = f"LinkedIn profile URL: {linkedin_url}\nUser request: {user_query}"
    response = await agent.run(prompt)
    raw = str(response).strip()

    # strip accidental markdown
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        import json
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON, returning raw string")
        return {"raw": raw}