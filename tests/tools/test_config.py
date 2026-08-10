from dotenv import load_dotenv

from langchain.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from tools.config import get_embeddings, get_llm

load_dotenv()


def test_get_embeddings():
    embeddings = get_embeddings()
    assert embeddings is not None, "Embeddings should not be None"
    assert isinstance(embeddings, Embeddings)


def test_get_llm():
    llm = get_llm()
    assert llm is not None, "LLM should not be None"
    assert isinstance(llm, BaseChatModel)

