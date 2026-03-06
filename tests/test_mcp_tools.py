import asyncio
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from src.config.settings import settings

async def main():
    client = BasicMCPClient(settings.mcp_server_url)
    spec = McpToolSpec(client=client)
    tools = await spec.to_tool_list_async()
    for tool in tools:
        print(tool.metadata.name)
        print(tool.metadata.description)
        print("---")

asyncio.run(main())