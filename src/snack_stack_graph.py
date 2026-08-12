from multi_key_dict import multi_key_dict

from langchain_chroma import Chroma
from langchain.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver



from agents.context_schema import ContextSchema
from agents.state import GRAPH_END, MENU_AGENT, MENU_AGENT_TOOLS, ORCHESTRATOR_AGENT, ORDER_AGENT, ORDER_AGENT_TOOLS, SYNTHESIZER_AGENT, SnackStackState
from agents.orchestrator import dispatch_to_agents, orchestrator_node
from agents.menu_agent import menu_agent_node, menu_agent_should_continue
from agents.order_agent import order_agent_node, order_agent_should_continue
from agents.synthesizer import synthesizer_node

MENU_AGENT_NODE = "menu_agent_node"
MENU_AGENT_TOOL_NODE = "menu_agent_tool_node"
ORDER_AGENT_NODE = "order_agent_node"
ORDER_AGENT_TOOL_NODE = "order_agent_tool_node"
SYNTHESIZER_AGENT_NODE = "synthesizer_agent_node"



def build_graph(orders: multi_key_dict, menu_collection: Chroma, llm: BaseChatModel):
  context = ContextSchema(orders=orders, menu_collection=menu_collection, llm=llm)
  builder = StateGraph(SnackStackState)

  #create tool nodes
  menu_agent_tool_node = ToolNode(tools=context.menu_tools)
  order_agent_tool_node = ToolNode(tools=context.orders_tools)

  #add nodes
  builder.add_node(ORCHESTRATOR_AGENT, orchestrator_node)
  builder.add_node(MENU_AGENT_NODE, menu_agent_node)
  builder.add_node(MENU_AGENT_TOOL_NODE, menu_agent_tool_node)
  builder.add_node(ORDER_AGENT_NODE, order_agent_node)
  builder.add_node(ORDER_AGENT_TOOL_NODE, order_agent_tool_node)
  builder.add_node(SYNTHESIZER_AGENT_NODE, synthesizer_node)

  #add edges
  #orchestrator edges
  builder.add_edge(START, ORCHESTRATOR_AGENT)
  builder.add_conditional_edges(ORCHESTRATOR_AGENT, dispatch_to_agents, {
    MENU_AGENT: MENU_AGENT_NODE,
    ORDER_AGENT: ORDER_AGENT_NODE
  })
  
  #menu agent edges
  builder.add_conditional_edges(MENU_AGENT_NODE, menu_agent_should_continue, {
    MENU_AGENT_TOOLS: MENU_AGENT_TOOL_NODE,
    SYNTHESIZER_AGENT: SYNTHESIZER_AGENT_NODE,
    GRAPH_END: END
  })
  builder.add_edge(MENU_AGENT_TOOL_NODE, MENU_AGENT_NODE)

  #order agent edges
  builder.add_conditional_edges(ORDER_AGENT_NODE, order_agent_should_continue, {
    ORDER_AGENT_TOOLS: ORDER_AGENT_TOOL_NODE,
    SYNTHESIZER_AGENT: SYNTHESIZER_AGENT_NODE,
    GRAPH_END: END
  })
  builder.add_edge(ORDER_AGENT_TOOL_NODE, ORDER_AGENT_NODE)
  builder.add_edge(SYNTHESIZER_AGENT_NODE, END)

  #add memory for persistence checkpointer and troubleshooting
  memory = InMemorySaver()
  return builder.compile(checkpointer=memory), context

