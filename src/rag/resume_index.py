import tempfile 
from pathlib import Path 
import json

import chromadb 
from llama_index.core import VectorStoreIndex,StorageContext,Settings,PromptTemplate
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import SentenceSplitter

from src.config.settings import settings 
from src.config.logger import logger 
from src.llm.llm_interface import get_llm

_chroma_client = chromadb.PersistentClient("./chroma_db")
RESUME_COLLECTION = "resume"
JOB_COLLECTION = "job_posting"

def build_chroma_index(nodes: list,collection_name: str) -> VectorStoreIndex:
    """
    Drops any existing collection with the same name and re-indexes nodes.
    This gives us a clean slate every time a new resume or job is loaded,
    which avoids stale data from a previous session.

    Args:
        nodes:           TextNode list from resume_processing or job_processing.
        collection_name: ChromaDB collection identifier.

    Returns:
        A VectorStoreIndex backed by ChromaDB.
    """

    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass 

    chroma_collection = _chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes,storage_context=storage_context)
    logger.info(f"Chromadb indexed {len(nodes)} nodes -> collection: {collection_name}")
    return index


def build_resume_tool(nodes: list, candidate_name: str) -> QueryEngineTool:
    """
    Indexes resume nodes into ChromaDB and wraps the query engine
    as a QueryEngineTool named 'resume_search'.

    This is the counterpart to build_job_tool() — both follow the same
    pattern so the two tools are symmetric and consistent for the agents.

    Args:
        nodes:          TextNode list from process_resume().
        candidate_name: Used in the tool description so the agent knows
                        whose resume it is querying.

    Returns:
        QueryEngineTool named "resume_search".
    """
    llm = get_llm()
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_id)

    resume_prompt = PromptTemplate(settings.resume_qa_template)
    index = build_chroma_index(nodes, RESUME_COLLECTION)

    engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.similarity_top_k,
        text_qa_template=resume_prompt,
    )

    tool = QueryEngineTool.from_defaults(
        query_engine=engine,
        name="resume_search",
        description=(
            f"Search {candidate_name}'s resume for specific information. "
            "Use this to retrieve: skills, work experience, education, "
            "certifications, projects, and achievements. "
            "Always call this tool before making any claims about the candidate."
        ),
    )

    logger.info(f"[ResumeIndex] resume_search tool built for: {candidate_name}")
    return tool


def build_job_tool(job_nodes: list,job_title: str) -> QueryEngineTool:
    """
    Indexes job posting nodes and wraps the engine as a QueryEngineTool.

    Args:
        job_nodes: Chunked nodes from process_job_posting().
        job_title: Used in tool description.

    Returns:
        QueryEngineTool named "job_search".
    """
    llm = get_llm()
    Settings.llm = llm 
    Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_id)

    job_prompt = PromptTemplate(settings.job_qa_template)
    index = build_chroma_index(job_nodes,JOB_COLLECTION)

    engine = index.as_query_engine(llm=llm,
                                   similarity_top_k=settings.similarity_top_k,
                                   text_qa_template=job_prompt)
    
    tool = QueryEngineTool.from_defaults(query_engine=engine,
                                         name="job_search",
                                         description=(f"Search the job posting for '{job_title}' for specific requirements. "
                                                      "Use this to retrieve: required skills, preferred skills, responsibilities, "
                                                      "required experience years, required education, and seniority level. "
                                                      "Always call this tool before making any claims about the job requirements."),)
    
    logger.info(f"[JobIndex] job_search tool built for: {job_title}")
    return tool

def process_job_posting(job_data:dict) -> list:
    """
    Converts a structured job posting dict into LlamaIndex nodes
    using the same SentenceSplitter pattern as data_processing.py.

    We write the dict to a temp JSON file and use the same
    JSONReader approach so the chunking is consistent.

    Args:
        job_data: Parsed dict from job_client.fetch_job_posting().

    Returns:
        List of TextNode objects ready for _build_chroma_index().
    """
    if not job_data:
        raise ValueError("job_data is empty nothing to process")
    
    reader = JSONReader(levels_back=0,clean_json=True)

    with tempfile.NamedTemporaryFile(mode='w',suffix=".json",delete=False,encoding="utf-8") as tmp:
        json.dump(job_data,tmp,indent=2)
        tmp_path = Path(tmp.name)

    documents = reader.load_data(input_file=tmp_path)
    tmp_path.unlink()
    
    splitter = SentenceSplitter(chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap)

    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes: 
        node.metadata.update({"source":"job_posting",
                              "job_title":job_data.get("job_title","unknown"),
                              "company":job_data.get("company","unknown")})

    logger.info(
        f"[JobIndex] Processed {len(nodes)} nodes for "
        f"{job_data.get('job_title')} @ {job_data.get('company')}"
    )
    return nodes 