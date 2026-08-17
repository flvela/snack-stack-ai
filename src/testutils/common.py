"""Common utils used for unit tests"""
from langgraph.runtime import Runtime

from agents.context_schema import ContextSchema
from tools.config import get_llm


def init_test_runtime():
  """test runtime initialization for LangGraph agents"""
  context = ContextSchema(orders=None, menu_collection=None, llm=get_llm())
  return Runtime(context=context)
