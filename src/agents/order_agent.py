from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agents.context_schema import ContextSchema
from agents.state import MESSAGES_FIELD, ORDER_AGENT_OUTPUT_FIELD, USER_INPUT_FIELD, ORDER_AGENT_TOOLS, GRAPH_END, SnackStackState


order_instructions = """
You are an order agent. Your job is to answer order questions about orders for our restaurant.
You can use *search_order_catalog* tool to fetch order data by order_id, email and tracking number.
Output rules:
- Provide order(s) information based on the data retrieved using the *search_order_catalog* only.
- If order(s) are not found. Say that the order number, email or tracking number has no associated orders.
- order id and tracking id are unique.
- email id can return multiple orders. In this case you can display all orders and ask which one they want more details on.
"""

def order_agent_node(state: SnackStackState, runtime: Runtime[ContextSchema]) -> SnackStackState:
  """ order agent specializing on answering order questions """
  if not state.get(MESSAGES_FIELD):
    messages = [
      SystemMessage(content=order_instructions),
      HumanMessage(content=state[USER_INPUT_FIELD])
    ]
  else:
    messages = state.get(MESSAGES_FIELD)

  result = runtime.context.orders_llm.invoke(messages)
  if result.tool_calls:
    return {MESSAGES_FIELD: [*messages, result] if not state.get(MESSAGES_FIELD) else [result]}
  else:
    return {ORDER_AGENT_OUTPUT_FIELD: result.content, MESSAGES_FIELD: [result]}

def order_agent_should_continue(state: SnackStackState) -> str:
  if MESSAGES_FIELD in state and len(state[MESSAGES_FIELD]) > 0:
    last_msg = state[MESSAGES_FIELD][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
      return ORDER_AGENT_TOOLS
  return GRAPH_END