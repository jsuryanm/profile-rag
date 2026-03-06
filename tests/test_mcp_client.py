# test_mcp_client.py
import asyncio
from mcp_client.linkedin_client import fetch_profile_agent

async def main():
    linkedin_url = "https://www.linkedin.com/in/satyanadella/"
    query = "Get only the main profile page for this person, with sections experience and education"
    
    print("Connecting to LinkedIn MCP server...")
    result = await fetch_profile_agent(query, linkedin_url)
    print("\n--- RESULT ---")
    print(result)

asyncio.run(main())