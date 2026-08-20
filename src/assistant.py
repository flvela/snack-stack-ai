"""Defines the Snack stack AI assistant"""
import shutil
from typing import List
import uuid

from langchain_core.documents import Document
from langgraph.graph.state import Command, CompiledStateGraph
from multi_key_dict import multi_key_dict


from agents.context_schema import ContextSchema
from agents.state import FINAL_RESPONSE_FIELD
from snack_stack_graph import build_graph
from tools.config import Config
from tools.menu import MENU_COLLECTION_NAME
from tools.vector_store import VectorStore

ORDERS_DIRECTORY = "data/orders.json"
PERSISTENT_STORE_DIRECTORY = "data/chroma_store"
MENU_DIRECTORY = "data/menu.json"
DEFAULT_ANSWER = "Sorry I could not process your request"


class SnackStackAssistant:
  """snack stack AI assistant class. Defines ask and AI operations"""
  graph: CompiledStateGraph
  context: ContextSchema
  vector_store: VectorStore
  thread_id: str
  orders: multi_key_dict
  menu: List[Document]
  is_interrupted: bool

  def __init__(self,
               orders: multi_key_dict,
               menu: List[Document],
               config: Config = Config()):
    """Constructor for SnackStackAssistant"""
    self.orders = orders
    self.documents = menu
    self.vector_store = VectorStore(persist_directory=PERSISTENT_STORE_DIRECTORY,
                                    collection_name=MENU_COLLECTION_NAME)
    menu_collection = self.vector_store.get_create_collection(menu, config=config)
    self.graph, self.context = build_graph(orders=orders,
                                           menu_collection=menu_collection,
                                           llm=config.get_llm())
    self.thread_id = str(uuid.uuid4())
    self.is_interrupted = False

  def reset(self):
    """resets the graph state and conversation"""
    self.thread_id = str(uuid.uuid4())
    self.is_interrupted = False

  def ask(self, query: str):
    """invokes snack stack graph with user query and handles human-in-the-loop (HITL) interrupts.
    To start a new conversation use reset() method

    Args:
      query: the original user query to send to the graph
    """
    if self.is_interrupted:
      graph_input = Command(resume=query)
      self.is_interrupted = False
    else:
      graph_input = {
        "user_input": query,
        "messages": [],
        "output": "",
        "route": ""}

    config = {"configurable": {"thread_id": self.thread_id}}
    result = self.graph.invoke(graph_input, config, context=self.context)
    print(f"result: {result}")
    snapshot = self.graph.get_state(config)
    print(f"snapshot: {snapshot}")
    for task in snapshot.tasks:
      if not getattr(task, "interrupts", None):
        continue
      for intr in task.interrupts:
        prompt = intr.value
        self.is_interrupted = True
        return prompt

    return result.get(FINAL_RESPONSE_FIELD, DEFAULT_ANSWER)

  def shutdown(self):
    """Closes connection to vector store and cleans up data"""
    self.vector_store.delete_collection()
    self.vector_store.close()
    shutil.rmtree(PERSISTENT_STORE_DIRECTORY)
