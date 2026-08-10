from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agents.context_schema import ContextSchema
from agents.state import ROUTE_FIELD, USER_INPUT_FIELD, SnackStackState

orchestrator_instructions = """
You are an orchestrator agent and your job is to classify each user query.
Here are the different ways to classify:
1. If user is asking for food/menu queries, respond with menu_agent.
2. If user is asking about order status or order tracking, respond with order_agent.
3. If user is asking about both order status and food/menu queries, respond with both.
4. If user intent is not clear, respond with menu_agent.

Respond only with: menu_agent, order_agent, both
"""

def orchestrator_node(state: SnackStackState, runtime: Runtime[ContextSchema]) -> SnackStackState:
  """Orchestrator Agent node that decides how to route traffic to either to Menu Agent, Order Agent or both"""
  messages = [
    SystemMessage(content=orchestrator_instructions),
    HumanMessage(content=state[USER_INPUT_FIELD])
  ]

  result = runtime.context.llm.invoke(messages)
  return {ROUTE_FIELD: result.content, "messages":[HumanMessage(content=state[USER_INPUT_FIELD])]}

def route_decision(state: SnackStackState) -> str:
  return state[ROUTE_FIELD]