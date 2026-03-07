import asyncio
from mcp_client.linkedin_client import fetch_profile_agent
from src.processing.data_processing import process_profile
from src.rag.query_engine import build_router_query_engine, query_profile


async def main():
    linkedin_url = "https://www.linkedin.com/in/satyanadella/"  

    print(f"\nFetching profile: {linkedin_url}")
    profile_data = await fetch_profile_agent(
        user_query="Fetch the full LinkedIn profile including experience and education",
        linkedin_url=linkedin_url
    )

    if not profile_data or "raw" in profile_data:
        print("Failed to fetch profile. Check that linkedin-mcp-server is running.")
        print(f"Raw response: {profile_data}")
        return

    subject_name = profile_data.get("name", "Unknown")
    print(f"Successfully fetched profile for: {subject_name}")
    print(f"Profile data: {profile_data}\n")

    nodes = process_profile(profile_data, metadata={"source": "linkedin", "url": linkedin_url})
    router = build_router_query_engine(nodes, subject_name=subject_name)

    # Should route to profile_qa
    simple_questions = [
        "What is his current role?",
        "Where did he study?",
        "What company does he work at?",
    ]

    # Should route to profile_report
    complex_questions = [
        "Give me a full summary of this person's career",
        "What are some good icebreaker questions I could ask him?",
        "Give me networking tips for connecting with this person",
    ]

    print("\n--- SIMPLE QUESTIONS (expect: profile_qa) ---")
    for q in simple_questions:
        print(f"\nQ: {q}")
        answer = await query_profile(router, q, subject_name=subject_name)
        print(f"A: {answer}")

    print("\n--- COMPLEX QUESTIONS (expect: profile_report) ---")
    for q in complex_questions:
        print(f"\nQ: {q}")
        answer = await query_profile(router, q, subject_name=subject_name)
        print(f"A: {answer}")


asyncio.run(main())