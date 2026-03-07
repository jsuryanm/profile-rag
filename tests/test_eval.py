import asyncio
import json
from src.rag.eval import evaluate_router
from mcp_client.linkedin_client import fetch_profile_agent
from src.processing.data_processing import process_profile
from src.rag.query_engine import build_router_query_engine

async def main():
    # 1. Fetch profile
    profile_data = await fetch_profile_agent(
        user_query="Fetch the full LinkedIn profile including experience and education",
        linkedin_url="https://www.linkedin.com/in/satyanadella/"
    )

    # 2. Guard against failed fetch before doing anything else
    if not profile_data or "raw" in profile_data:
        print(" Profile fetch failed — LinkedIn likely rate limited.")
        print("Raw response:", profile_data.get("raw", "empty"))
        print("Wait a few minutes and try again.")
        return

    subject_name = profile_data.get("name", "Unknown")
    if subject_name == "Unknown":
        print(" Profile returned but name is missing — data may be incomplete.")
        print(json.dumps(profile_data, indent=2))
        return

    print(f" Profile fetched for: {subject_name}")

    # 3. Build index
    nodes = process_profile(profile_data)
    router = build_router_query_engine(nodes, subject_name=subject_name)

    # 4. Run eval
    results = await evaluate_router(router, subject_name=subject_name)

    print("\n========== EVAL RESULTS ==========")
    for metric, eval_results in results.items():
        print(f"\n--- {metric.upper()} ---")
        for i, r in enumerate(eval_results):
            print(f"Q{i+1}: passing={r.passing} | score={r.score}")
            if r.feedback:
                print(f"     feedback: {r.feedback}")

if __name__ == "__main__":
    asyncio.run(main())