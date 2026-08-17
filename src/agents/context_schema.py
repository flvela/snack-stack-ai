"""Shared context schema definition for LangGraph Runtime."""
from dataclasses import dataclass
from typing import List

from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel
from langchain_chroma import Chroma

from multi_key_dict import multi_key_dict

from tools.common import get_user_input
from tools.menu import search_menu_catalog
from tools.orders import search_order_catalog


@dataclass
class ContextSchema:
  """Graph Runtime Context schema"""
  # orders documents
  orders: multi_key_dict
  # menu items vector store
  menu_collection: Chroma
  # the llm used to process/generate data
  llm: BaseChatModel
  # the menu llm used to process menu data
  menu_llm: BaseChatModel
  # the menu tools
  menu_tools: List[BaseTool]
  # the orders llm used to process order status queries
  orders_llm: BaseChatModel
  # the orders tools
  orders_tools: List[BaseTool]

  def __init__(self, orders: multi_key_dict, menu_collection: Chroma, llm: BaseChatModel):
    self.orders = orders
    self.menu_collection = menu_collection
    self.llm = llm
    self.menu_tools = [search_menu_catalog]
    self.menu_llm = llm.bind_tools(self.menu_tools)
    self.orders_tools = [search_order_catalog, get_user_input]
    self.orders_llm = llm.bind_tools(self.orders_tools)
