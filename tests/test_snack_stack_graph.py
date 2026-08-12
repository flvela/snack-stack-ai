import os
from dotenv import load_dotenv
from langgraph.graph import END, START

from agents.state import GRAPH_END, MENU_AGENT, MENU_AGENT_TOOLS, ORCHESTRATOR_AGENT, ORDER_AGENT, ORDER_AGENT_TOOLS, SYNTHESIZER_AGENT, SnackStackState
from snack_stack_graph import MENU_AGENT_NODE, MENU_AGENT_TOOL_NODE, ORDER_AGENT_NODE, ORDER_AGENT_TOOL_NODE,SYNTHESIZER_AGENT_NODE, build_graph
from tools.config import get_llm
from tools.menu import MENU_COLLECTION_NAME, load_menu_documents
from tools.vector_store import create_vector_store
from tools.orders import load_orders_documents


load_dotenv()


def test_build_graph():
  print("\n")
  expected_nodes = [START, ORCHESTRATOR_AGENT, MENU_AGENT_NODE, MENU_AGENT_TOOL_NODE, ORDER_AGENT_NODE, ORDER_AGENT_TOOL_NODE, SYNTHESIZER_AGENT_NODE, END]
  expected_edges = {
    START: {
      ORCHESTRATOR_AGENT: (False,None)
    },
    ORCHESTRATOR_AGENT: {
      ORDER_AGENT_NODE: (True, ORDER_AGENT),
      MENU_AGENT_NODE: (True, MENU_AGENT)
    },
    MENU_AGENT_NODE: {
      MENU_AGENT_TOOL_NODE: (True, MENU_AGENT_TOOLS),
      SYNTHESIZER_AGENT_NODE: (True, SYNTHESIZER_AGENT),
      END: (True, GRAPH_END)
    },
    MENU_AGENT_TOOL_NODE : {
      MENU_AGENT_NODE: (False, None)
    },
    ORDER_AGENT_NODE: {
      ORDER_AGENT_TOOL_NODE: (True, ORDER_AGENT_TOOLS),
      SYNTHESIZER_AGENT_NODE: (True, SYNTHESIZER_AGENT),
      END: (True, GRAPH_END)
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
  menu_store = create_vector_store(persist_directory, documents, collection_name=MENU_COLLECTION_NAME, overwrite=True)
  llm = get_llm()
  graph, context = build_graph(orders=orders, menu_collection=menu_store, llm=llm)
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
  assert context.menu_collection == menu_store
  assert context.llm == llm