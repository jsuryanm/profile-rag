from pydantic_settings import BaseSettings 
from pydantic import Field, ConfigDict

# Base Settings allows fields to be overriden by env variables

class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(...,validation_alias="OPENAI_API_KEY")
    llm_model_id: str = "gpt-5-mini"
    model_config = ConfigDict(env_file=".env",extra="ignore")
    temperature: float = 0.0 

    # embeddings
    embedding_model_id: str = "text-embedding-3-small"

    # RAG 
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 2

    # MCP SERVER
    mcp_server_url: str = "http://127.0.0.1:8080/mcp"
    mcp_server_port: int = 8080

    # mock data safety fallback 
    mock_data_url: str = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
        "/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"
    )

    # Prompt template
    initial_facts_template: str = """
    You are an AI assistant. Using ONLY the context below, list 3 interesting about this person's career or education.

    Context: {context_str}

    Answer in detail using only the provided context. 
    """
    
    user_question_template: str = """
    You are an AI assistant. Answer the question using ONLY the content below.
    If the answer is not in the context, say:
    "I don't know. The information is not available on the LinkedIn page."
    
    Context: {context_str}
    Question: {query_str}
    """

    report_template: str = """
    You are a professional career analyst. Using ONLY the context below,
    generate a complete structured profile report.

    For icebreaker_questions: write 3 specific, thoughtful questions
    you could ask this person to start a meaningful conversation.
    For networking_tips: suggest how best to approach and connect with them.

    Context: {context_str}
    """

    resume_qa_template: str = """
    You are an expert resume analyst. Answer the question using ONLY the resume content below.
    Focus on extracting specific, factual information.
    If the information is not in the resume, say: "This information is not in the resume."

    Resume Content: {context_str}
    Question: {query_str}
    """

    job_qa_template: str = """
    You are an expert at analysing job postings. Answer the question using ONLY the job posting content below.
    Extract precise requirements, skills, and qualifications as stated.
    If the information is not in the job posting, say: "This is not specified in the job posting."

    Job Posting Content: {context_str}
    Question: {query_str}
    """






     


# Singleton — import this everywhere instead of re-instantiating
settings = Settings()
# print("OPENAI KEY:", settings.openai_api_key[:10])