import asyncio
from src.processing.data_processing import process_profile
from src.rag.query_engine import build_query_engine, query_profile

async def main():
    profile = {
        "name": "Satya Nadella",
        "headline": "Chairman and CEO at Microsoft",
        "location": "Redmond, Washington, United States",
        "current_role": {"title": "Chairman and CEO", "company": "Microsoft", "duration": "Feb 2014 - Present"},
        "experience": [{"title": "Chairman and CEO", "company": "Microsoft", "duration": "Feb 2014 - Present"}],
        "education": [{"school": "University of Chicago Booth School of Business", "degree": "MBA", "years": "1994-1996"}]
    }

    nodes = process_profile(profile, metadata={"source": "linkedin"})
    query_engine = build_query_engine(nodes)

    questions = [
        "What is his current role?",
        "Where did he study?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = await query_profile(query_engine, q, subject_name="Satya Nadella")
        print(f"A: {answer}")

asyncio.run(main())