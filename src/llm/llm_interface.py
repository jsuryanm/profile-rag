from llama_index.llms.groq import Groq 
from src.config.settings import settings
from src.config.logger import logger 

_llm_instance = None 

def get_llm(temperature: float = None) -> Groq:
    """
    Returns a cached Groq LLM instance.
    Pass temperature to override the default from settings  
    """

    global _llm_instance
    temp = temperature if temperature is not None else settings.temperature

    if _llm_instance is None:
        logger.info(f"Initializing Groq LLM: {settings.llm_model_id} temperature:{settings.temperature}")
        _llm_instance = Groq(model=settings.llm_model_id,
                             temperature=temp,
                             api_key=settings.groq_api_key)
    
    return _llm_instance

def reset_llm() -> None:
    """Clears the cached LLM instance call this for 
    changing model id at runtime"""

    global _llm_instance 
    _llm_instance = None 
    logger.info("LLM instance reset.")