"""Configures the Embeddings and LLM based on environment config"""
import os

from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings.base import init_embeddings
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction, OpenAIEmbeddingFunction


load_dotenv()  # Load environment variables from .env file

SUPPORTED_EMBEDDINGS_PROVIDERS = ["huggingface", "openai"]


def get_langchain_embeddings(provider=os.getenv("EMBEDDINGS_MODEL_PROVIDER"),
                             model=os.getenv("EMBEDDINGS_MODEL"),
                             api_key=os.getenv("EMBEDDINGS_MODEL_API_KEY")) -> Embeddings:
  """
  Get the langchain embeddings model from arguments or environment variables.

  Returns:
    Embeddings: The langchain embeddings model
  """
  if provider == "huggingface":
    kwargs = {"model_kwargs": {"token": api_key}}
  elif provider == "openai":
    kwargs = {"openai_api_key": api_key}
  else:
    raise NotImplementedError(
       (f"Embedddings model provider {provider} is not one of the"
        f" supported providers: {SUPPORTED_EMBEDDINGS_PROVIDERS}"))
  return init_embeddings(provider=provider, model=model, **kwargs)


def get_chromadb_embeddings(provider=os.getenv("EMBEDDINGS_MODEL_PROVIDER"),
                            model=os.getenv("EMBEDDINGS_MODEL"),
                            api_key=os.getenv("EMBEDDINGS_MODEL_API_KEY")) -> Embeddings:
  """
  Get the chromadb embeddings model from arguments or environment variables.

  Returns:
    Embeddings: The embeddings model based on environment variables
  """
  if provider == "huggingface":
    return HuggingFaceEmbeddingFunction(api_key=api_key, model_name=model)

  if provider == "openai":
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

  raise NotImplementedError(
    (f"Embedddings model provider {provider} is not one of the"
     f" supported providers: {SUPPORTED_EMBEDDINGS_PROVIDERS}"))


def get_llm(model=os.getenv("MODEL"),
            model_provider=os.getenv("MODEL_PROVIDER"),
            api_key=os.getenv("MODEL_API_KEY")) -> BaseChatModel:
  """
  GET the llm chat model from arguments or environment variables.
  Args:
    model: the name of the model to use
    model_provider: the name of the model provider. Must be supported by langchain init_chat_model parameters
    api_key: the model provider api key.
  """
  return init_chat_model(model=model, model_provider=model_provider, temperature=0, api_key=api_key)
