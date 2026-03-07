import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config.settings import settings
from src.config.logger import logger
from src.llm.llm_interface import get_llm

_chroma_client = chromadb.PersistentClient("./chroma_db")


def _build_index(nodes: list, collection_name: str = "linkedin_profile") -> VectorStoreIndex:
    """Indexes nodes into ChromaDB and returns the VectorStoreIndex."""
    try:
        _chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    chroma_collection = _chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    logger.info(f"Indexed {len(nodes)} nodes into chromadb collection: {collection_name}")
    return index


def build_router_query_engine(nodes: list, subject_name: str = "this person") -> RouterQueryEngine:
    """
    Builds a RouterQueryEngine with two tools:
    - Simple QA: for direct factual questions
    - Report generator: for summaries, icebreakers, networking tips
    The LLMSingleSelector picks the right tool based on the question.
    """
    llm = get_llm()
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_id)

    index = _build_index(nodes)

    # Tool 1 — direct factual QA
    qa_prompt = PromptTemplate(settings.user_question_template)
    qa_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.similarity_top_k,
        text_qa_template=qa_prompt
    )

    # Tool 2 — structured report generation
    report_prompt = PromptTemplate(settings.report_template)
    report_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.similarity_top_k,
        text_qa_template=report_prompt
    )

    # QueryEngineTool is a wrapper that turns a query engine into a tool 
    # that an agent can call
    qa_tool = QueryEngineTool.from_defaults(
        query_engine=qa_engine,
        name="profile_qa",
        description=f"""Use this for specific factual questions about {subject_name}.
        Examples: current role, education, work history, skills, location, job titles.
        Best for direct single-fact questions that have a clear answer in the profile."""
    )

    report_tool = QueryEngineTool.from_defaults(
        query_engine=report_engine,
        name="profile_report",
        description=f"""Use this to generate a structured report about {subject_name}.
        Examples: full career summary, icebreaker questions, networking tips, 
        professional overview, or any request asking for multiple insights at once."""
    )

    # RouterQueryEngine is a meta query engine that decides which tool should answer a question

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[qa_tool, report_tool],
        verbose=True  # logs which tool was selected — useful for debugging
    )

    logger.info(f"RouterQueryEngine built for: {subject_name}")
    return router


def build_agentic_rag(router: RouterQueryEngine,
                      subject_name: str = "this person",
                      memory: ChatMemoryBuffer =  None) -> FunctionAgent:
    """
    Wraps the RouterQueryEngine as a single tool inside a FunctionAgent.
    The agent adds memory for follow-up questions.
    The router handles tool selection internally.
    """
    llm = get_llm()

    router_tool = QueryEngineTool.from_defaults(
        query_engine=router,
        name="profile_router",
        description=f"""The main tool for answering any question about {subject_name}'s 
        LinkedIn profile. Handles both simple factual questions and complex requests 
        like reports, icebreakers, and networking tips. Always use this tool."""
    )

    if memory is None:
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    agent = FunctionAgent(
        tools=[router_tool],
        llm=llm,
        memory=memory,
        system_prompt=f"""You are a professional profile research assistant for {subject_name}.
        You have one tool: profile_router. 
        Rules:
        - ALWAYS call profile_router once before answering
        - NEVER call the tool more than once per question
        - Return the tool result as your final answer immediately
        - If information is not available, say so clearly"""
    )

    logger.info(f"FunctionAgent with RouterQueryEngine built for: {subject_name}")
    return agent


async def query_profile(router: RouterQueryEngine, question: str, subject_name: str = None) -> str:
    """Direct router query — no agent, no memory. For stateless single questions."""
    if subject_name:
        question = f"Regarding {subject_name}: {question}"
    logger.info(f"Router query: {question}")
    response = router.query(question)
    return str(response)


async def query_profile_agentic(agent: FunctionAgent, question: str) -> str:
    """Agentic query — uses memory for follow-up questions."""
    logger.info(f"Agentic query: {question}")
    response = await agent.run(
        question,
        max_iterations=5,
        early_stopping_method="generate"
    )
    return str(response)

