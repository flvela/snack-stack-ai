"""Defines the Orchestrator agent for LangGraph"""
from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Send

from agents.context_schema import ContextSchema
from agents.state import (
  FINAL_RESPONSE_FIELD,
  MENU_AGENT_MESSAGES_FIELD,
  MENU_AGENT_OUTPUT_FIELD,
  MESSAGES_FIELD,
  ORDER_AGENT_MESSAGES_FIELD,
  ORDER_AGENT_OUTPUT_FIELD,
  REQUIRES_SYNTHESIS_FIELD,
  TASKS_FIELD,
  USER_INPUT_FIELD,
  OrchestratorResult,
  SnackStackState,
  state_to_string
)

ORCHESTRATOR_INSTRUCTIONS = """
You are an orchestrator agent and your job is to classify each user query
and dispatch to menu_agent, order_agent or both.

Here are the different ways to dispacth:
1. If user is asking for food/menu queries send to menu_agent.
2. If user is asking about order status or order tracking send to order_agent.
3. If user is asking about both order status and food/menu queries send to menu_agent and order agent.
4. If user intent is not clear send to menu_agent.
"""


def orchestrator_node(state: SnackStackState, runtime: Runtime[ContextSchema]) -> SnackStackState:
  """Orchestrator Agent node that decides how to route traffic
  to either to Menu Agent, Order Agent or both"""
  history = state.get(MESSAGES_FIELD, [])[-3:]
  messages = [
    SystemMessage(content=ORCHESTRATOR_INSTRUCTIONS),
    *history,
    HumanMessage(content=state[USER_INPUT_FIELD])
  ]
  print(f"\n###orchestrator_node###\n{state_to_string(state)}")
  result = runtime.context.llm.with_structured_output(OrchestratorResult).invoke(messages)
  return {TASKS_FIELD: result.tasks,
          REQUIRES_SYNTHESIS_FIELD: len(result.tasks) > 1,
          USER_INPUT_FIELD: state[USER_INPUT_FIELD]}


def dispatch_to_agents(state: SnackStackState):
  """Uses the Send API to dispatch to agent in parrallel if needed"""
  sends = []
  print(f"\n###dispatch_to_agents###\n{state_to_string(state)}")
  # reset worker state for downstream agents
  worker_state = {
    **state,
    MENU_AGENT_MESSAGES_FIELD: [],
    MENU_AGENT_OUTPUT_FIELD: "",
    ORDER_AGENT_MESSAGES_FIELD: [],
    ORDER_AGENT_OUTPUT_FIELD: "",
    FINAL_RESPONSE_FIELD: "",
  }
  for task in state.get(TASKS_FIELD):
    sends.append(Send(task.agent, worker_state))
  return sends
