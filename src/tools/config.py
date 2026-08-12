import os

from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings.base import init_embeddings
from langchain_core.embeddings import Embeddings


load_dotenv()  # Load environment variables from .env file

def get_embeddings(provider = os.getenv("EMBEDDINGS_MODEL_PROVIDER"), model=os.getenv("EMBEDDINGS_MODEL"), 
                   api_key=os.getenv("EMBEDDINGS_MODEL_API_KEY")) -> Embeddings:
    """
    Get the embeddings model from environment variables.

    Returns:
        Embeddsings: The embeddings model based on environment variables
    """
    if provider == "huggingface":
        kwargs = {"model_kwargs" : {"token" : api_key} }
    elif provider == "openai":
        kwargs = {"openai_api_key": api_key}
    else:
        raise NotImplementedError(f"Embedddings model support not available for {provider}")
        
    
    return init_embeddings(
            provider=provider,
            model=model,
            **kwargs
           )

def get_llm(model=os.getenv("MODEL"), model_provider=os.getenv("MODEL_PROVIDER"), api_key=os.getenv("MODEL_API_KEY")) -> BaseChatModel:
    """ 
    GET the llm chat model from environment variables.
    Args:
        model: the name of the model to use
        model_provider: the name of the model provider.  Must be supported by https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model#parameters
        api_key: the model provider api key. 
    """
    return init_chat_model(model=model, 
                      model_provider=model_provider, 
                      temperature=0,
                      api_key=api_key)