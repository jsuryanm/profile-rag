from llama_index.llms.openai import OpenAI
from src.config.settings import settings
from src.config.logger import logger


# Registry keyed by (model_id, temperature)
_llm_registry: dict[tuple[str, float], OpenAI] = {}


def get_llm(
    temperature: float | None = None,
    model_id: str | None = None,
) -> OpenAI:

    temp = temperature if temperature is not None else settings.temperature
    model = model_id if model_id is not None else settings.llm_model_id
    key = (model, temp)

    if key not in _llm_registry:
        logger.info(f"Initializing OpenAI LLM: {model} | temperature: {temp}")

        _llm_registry[key] = OpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temp,
            max_retries=3,
            reuse_client=True,
        )

    return _llm_registry[key]


def reset_llm(model_id: str | None = None, temperature: float | None = None) -> None:
    """Clear one specific entry or the entire registry."""

    if model_id or temperature is not None:
        key = (
            model_id or settings.llm_model_id,
            temperature if temperature is not None else settings.temperature,
        )
        _llm_registry.pop(key, None)
        logger.info(f"LLM instance reset for key: {key}")
    else:
        _llm_registry.clear()
        logger.info("All LLM instances reset.")