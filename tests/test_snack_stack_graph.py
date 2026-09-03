"""Unit tests for Snack Stack Graph"""
import shutil

from langgraph.graph import END, START

from agents import state
from snack_stack_graph import (
  MENU_AGENT_NODE,
  MENU_AGENT_TOOL_NODE,
  ORDER_AGENT_NODE,
  ORDER_AGENT_TOOL_NODE,
  SYNTHESIZER_AGENT_NODE,
  build_graph
  )
from tools.config import Config
from tools.menu import MENU_COLLECTION_NAME, load_menu_documents
from tools.vector_store import VectorStore
from tools.orders import load_orders_documents


def test_build_graph():
  """build_graph unit test"""
  expected_nodes = [START, state.ORCHESTRATOR_AGENT, MENU_AGENT_NODE,
                    MENU_AGENT_TOOL_NODE, ORDER_AGENT_NODE, ORDER_AGENT_TOOL_NODE,
                    SYNTHESIZER_AGENT_NODE, END]
  expected_edges = {
    START: {
      state.ORCHESTRATOR_AGENT: (False, None)
    },
    state.ORCHESTRATOR_AGENT: {
      ORDER_AGENT_NODE: (True, None),
      MENU_AGENT_NODE: (True, None)
    },
    MENU_AGENT_NODE: {
      MENU_AGENT_TOOL_NODE: (True, state.MENU_AGENT_TOOLS),
      SYNTHESIZER_AGENT_NODE: (True, state.SYNTHESIZER_AGENT),
      END: (True, state.GRAPH_END)
    },
    MENU_AGENT_TOOL_NODE: {
      MENU_AGENT_NODE: (False, None)
    },
    ORDER_AGENT_NODE: {
      ORDER_AGENT_TOOL_NODE: (True, state.ORDER_AGENT_TOOLS),
      SYNTHESIZER_AGENT_NODE: (True, state.SYNTHESIZER_AGENT),
      END: (True, state.GRAPH_END)
    },
    ORDER_AGENT_TOOL_NODE: {
      ORDER_AGENT_NODE: (False, None)
    },
    SYNTHESIZER_AGENT_NODE: {
      END: (False, None)
    }
  }
  orders = load_orders_documents("data/orders.json")
  persist_directory = 'data/chroma_store'
  documents = load_menu_documents('data/menu.json')
  vector_store = VectorStore(persist_directory=persist_directory,
                             collection_name=MENU_COLLECTION_NAME)
  menu_collection = vector_store.get_create_collection(documents)
  config = Config()
  llm = config.get_llm()
  graph, context = build_graph(orders=orders, menu_collection=menu_collection, llm=llm)
  try:
    assert list(graph.get_graph().nodes.keys()) == expected_nodes
    print(graph.get_graph().edges)
    for edge in graph.get_graph().edges:
      source_node = edge.source
      target_node = edge.target
      assert source_node in expected_edges
      assert target_node in expected_edges[source_node]
      attributes = expected_edges[source_node][target_node]
      assert attributes[0] == edge.conditional
      assert attributes[1] == edge.data
    assert context.orders == orders
    assert context.menu_collection == menu_collection
    assert context.llm == llm
  finally:
    # cleanup vector store if test passes or fails
    vector_store.delete_collection()
    vector_store.close()
    shutil.rmtree(persist_directory)
