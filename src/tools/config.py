import os

from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings.base import init_embeddings
from langchain_core.embeddings import Embeddings


load_dotenv()  # Load environment variables from .env file

def get_embeddings() -> Embeddings:
    """
    Get the embeddings model from environment variables.

    Returns:
        Embeddsings: The embeddings model based on environment variables
    """
    provider = os.getenv("EMBEDDINGS_MODEL_PROVIDER")
    if provider == "huggingface":
        kwargs = {"model_kwargs" : {"token" : os.getenv("EMBEDDINGS_MODEL_API_KEY")} }
    elif provider == "openai":
        kwargs = {"openai_api_key": os.getenv("EMBEDDINGS_MODEL_API_KEY")}
    else:
        raise NotImplementedError(f"Embedddings model support not available for {provider}")
        
    
    return init_embeddings(
            provider=provider,
            model=os.getenv("EMBEDDINGS_MODEL", "google/embeddinggemma-300m"),
            **kwargs
           )

def get_llm() -> BaseChatModel:
    """ 
    GET the llm chat model from environment variables.
    """
    return init_chat_model(model=os.getenv("MODEL"), 
                      model_provider=os.getenv("MODEL_PROVIDER"), 
                      temperature=0,
                      api_key=os.getenv("MODEL_API_KEY", "claude-haiku-4-5-20251001"))