import json 
import tempfile
from pathlib import Path 

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.json import JSONReader
from src.config.settings import settings 
from src.config.logger import logger 

def load_json_documents(profile_data: dict) -> list:
    """
    Writes profile dict to a temp file and loads it to LlamaIndex JSON Reader
    """
    reader = JSONReader(levels_back=0,clean_json=True)
    with tempfile.NamedTemporaryFile(mode="w",
                                     suffix=".json",
                                     delete=False,
                                     encoding="utf-8") as tmp:
        json.dump(profile_data,tmp,indent=2)
        tmp_path = Path(tmp.name)
    
    documents = reader.load_data(input_file=tmp_path)
    # converts json into LlamaIndex Document objects 
    
    tmp_path.unlink()

    logger.info(f"JSON Reader loaded {len(documents)} JSON documents from profile data")
    return documents

def chunk_documents(documents: list) -> list:
    """
    Splits documents into nodes using SentenceSplitter
    """ 
    if not documents:
        raise ValueError("No documents provided to chunk")
    
    splitter = SentenceSplitter(chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap)
    
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} chunks")
    logger.info(f"chunk_size = {settings.chunk_size} chunk_overlap = {settings.chunk_overlap}")
    return nodes 

def process_profile(profile_data: dict, metadata: dict = None) -> list:
    """Initiate the full pipeline"""
    if not profile_data:
        raise ValueError("profile data is empty nothing to process")
    
    documents = load_json_documents(profile_data=profile_data)

    if metadata:
        for doc in documents:
            doc.metadata.update(metadata)
    
    return chunk_documents(documents)