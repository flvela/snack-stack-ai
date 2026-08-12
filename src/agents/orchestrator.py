from typing import List

from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Send

from agents.context_schema import ContextSchema
from agents.state import MESSAGES_FIELD, REQUIRES_SYNTHESIS_FIELD, TASKS_FIELD, USER_INPUT_FIELD, OrchestratorResult, SnackStackState

orchestrator_instructions = """
You are an orchestrator agent and your job is to classify each user query
and dispatch to menu_agent, order_agent or both.

Here are the different ways to dispacth:
1. If user is asking for food/menu queries send to menu_agent.
2. If user is asking about order status or order tracking send to order_agent.
3. If user is asking about both order status and food/menu queries send to menu_agent and order agent.
4. If user intent is not clear send to menu_agent.
"""

def orchestrator_node(state: SnackStackState, runtime: Runtime[ContextSchema]) -> SnackStackState:
  """Orchestrator Agent node that decides how to route traffic to either to Menu Agent, Order Agent or both"""
  messages = [
    SystemMessage(content=orchestrator_instructions),
    HumanMessage(content=state[USER_INPUT_FIELD])
  ]

  result = runtime.context.llm.with_structured_output(OrchestratorResult).invoke(messages)
  return {TASKS_FIELD: result.tasks, 
          MESSAGES_FIELD:[HumanMessage(content=state[USER_INPUT_FIELD])],
          REQUIRES_SYNTHESIS_FIELD: len(result.tasks) > 1}

def dispatch_to_agents(state: SnackStackState) -> List[Send]:
  """Uses the Send API to dispatch to agent in parrallel if needed"""
  sends = []
  for task in state.get(TASKS_FIELD):
    sends.append(Send(task.agent, state))
  return sends