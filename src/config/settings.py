from pydantic_settings import BaseSettings 
from pydantic import Field, ConfigDict

# Base Settings allows fields to be overriden by env variables

class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(...,validation_alias="GROQ_API_KEY")
    llm_model_id: str = "llama-3.3-70b-versatile"
    model_config = ConfigDict(env_file=".env",extra="ignore")
    temperature: float = 0.0 

    # embeddings
    embedding_model_id: str = "BAAI/bge-small-en-v1.5"

    # RAG 
    chunk_size: int = 256
    chunk_overlap: int = 40
    similarity_top_k: int = 5 

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

     


# Singleton — import this everywhere instead of re-instantiating
settings = Settings()
