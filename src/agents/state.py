

from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

#orchestrator routes for graph edges
MENU_AGENT = "menu_agent"
ORDER_AGENT = "order_agent"
BOTH = "both"

#graph states for conditional edges
MENU_AGENT_TOOLS = "menu_agent_tools"
ORDER_AGENT_TOOLS = "order_agent_tools"
GRAPH_END = "end"

#state field names
MESSAGES_FIELD = "messages"
USER_INPUT_FIELD = "user_input"
ROUTE_FIELD = "route"
MENU_AGENT_OUTPUT_FIELD = "menu_agent_output"
ORDER_AGENT_OUTPUT_FIELD = "order_agent_output"
FINAL_RESPONSE_FIELD = "final_response"

class SnackStackState(TypedDict):
  """ Class defining the Graph State for the SnackStack assisstant. It will be updated by each node in the graph"""
  #conversation history
  messages: Annotated[List[BaseMessage], add_messages]
  #user query
  user_input: str
  #router decision
  route: Literal["menu_agent", "order_agent", "both"]
  #menu agent output
  menu_agent_output: str
  #order agent output
  order_agent_output: str
  #final response to user
  final_response: str