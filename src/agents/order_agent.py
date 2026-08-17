"""Order Agent definition"""
from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agents.context_schema import ContextSchema
from agents.state import (
  ORDER_AGENT_MESSAGES_FIELD,
  ORDER_AGENT_OUTPUT_FIELD,
  SYNTHESIZER_AGENT,
  USER_INPUT_FIELD,
  ORDER_AGENT_TOOLS,
  SnackStackState,
  state_to_string
)


ORDER_INSTRUCTIONS = """
You are an order agent. Your job is to answer order questions about orders for our restaurant.
You can use the following tools:
*search_order_catalog* tool to fetch order data by order_id, email and tracking number.
*get_user_input* tool to fetch the order key data from the user.
Tool selection rules:
- If user did not provide order_id, email or tracking number call *get_user_input* tool
- If user provides order_id, email or tracking number call *search_order_catalog* tool
Output rules:
- Provide order(s) information based on the data retrieved using the *search_order_catalog* only.
- If order(s) are not found. Say that the order number, email or tracking number has no associated orders.
- order id and tracking id are unique, email id can return multiple orders.
"""


def order_agent_node(state: SnackStackState, runtime: Runtime[ContextSchema]) -> SnackStackState:
  """ order agent specializing on answering order questions """
  print(f"\n###order_agent_node###\n{state_to_string(state)}")

  if not state.get(ORDER_AGENT_MESSAGES_FIELD):
    messages = [
      SystemMessage(content=ORDER_INSTRUCTIONS),
      HumanMessage(content=state[USER_INPUT_FIELD])
    ]
  else:
    messages = state.get(ORDER_AGENT_MESSAGES_FIELD)

  print(f"\n### messages {messages}")
  result = runtime.context.orders_llm.invoke(messages)
  print(f"\n### result {result}")
  if result.tool_calls:
    return {ORDER_AGENT_MESSAGES_FIELD: [*messages, result]}
  return {ORDER_AGENT_OUTPUT_FIELD: result.content}


def order_agent_should_continue(state: SnackStackState) -> str:
  """used to determine if order agent needs another turn. Condition on LangGraph edge"""
  print(f"\n###order_agent_should_continue###\n{state_to_string(state)}")
  if ORDER_AGENT_MESSAGES_FIELD in state and len(state[ORDER_AGENT_MESSAGES_FIELD]) > 0:
    last_msg = state[ORDER_AGENT_MESSAGES_FIELD][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
      return ORDER_AGENT_TOOLS
  return SYNTHESIZER_AGENT
