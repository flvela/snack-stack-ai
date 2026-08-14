"""Unit Tests for Config"""
import os

from dotenv import load_dotenv

from langchain.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction, OpenAIEmbeddingFunction

from tools.config import get_chromadb_embeddings, get_langchain_embeddings, get_llm

load_dotenv()


def test_get_langchain_embeddings():
  """test get_langchain_embeddings with default params. Ensures project is configured correctly"""
  embeddings = get_langchain_embeddings()
  assert embeddings is not None, "Embeddings should not be None"
  assert isinstance(embeddings, Embeddings)


def test_get_llm():
  """test get_llm with default params. Ensures project is configured correctly"""
  llm = get_llm()
  assert llm is not None, "LLM should not be None"
  assert isinstance(llm, BaseChatModel)

def test_get_chromadb_embeddings():
  """test get_chromadb_embeddings with default params. Ensures project is configured correctly"""
  provider = os.getenv("EMBEDDINGS_MODEL_PROVIDER")
  provider_to_type = {
    "huggingface": HuggingFaceEmbeddingFunction, 
    "openai": OpenAIEmbeddingFunction}
  embeddings = get_chromadb_embeddings()
  assert embeddings is not None, "Embeddings should not be None"
  assert isinstance(embeddings, provider_to_type[provider])
