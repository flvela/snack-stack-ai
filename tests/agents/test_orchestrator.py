from typing import List

import pytest
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime


from agents.context_schema import ContextSchema
from agents.state import MENU_AGENT, MESSAGES_FIELD, ORDER_AGENT, USER_INPUT_FIELD, TASKS_FIELD, REQUIRES_SYNTHESIS_FIELD, AgentTask, SnackStackState
from agents.orchestrator import orchestrator_node, dispatch_to_agents
from tools.config import get_llm

context = ContextSchema(orders=None, menu_collection=None, llm=get_llm())
runtime = Runtime(context=context)

BOTH = "both"

test_data = [
  ("Can you recommend some menu dishes to me", MENU_AGENT),
  ("What's the status of my order", ORDER_AGENT),
  ("Tell me about the reviews for Chicken tikka masala and my order status", BOTH),
  ("How's the weather in California", MENU_AGENT)
]

@pytest.mark.parametrize("user_input, expected_output", test_data)
def test_orchestrator_node(user_input: str, expected_output: str):
  state = SnackStackState()
  state[USER_INPUT_FIELD] = user_input
  result = orchestrator_node(state, runtime)
  print(result)
  tasks = result[TASKS_FIELD]

  agents = []
  for task in tasks:
    agents.append(task.agent)

  if expected_output == BOTH:
    assert len(agents) == 2
    assert MENU_AGENT in agents
    assert ORDER_AGENT in agents
    assert result[REQUIRES_SYNTHESIS_FIELD]
  else:
    assert len(agents) == 1
    assert expected_output in agents
    assert not result[REQUIRES_SYNTHESIS_FIELD]

  assert len(result[MESSAGES_FIELD]) == 1
  assert result[MESSAGES_FIELD][0].content == user_input

menu_agent_task =  AgentTask(agent=MENU_AGENT, description="Can you recommend some dishes to me")
order_agent_task =  AgentTask(agent=ORDER_AGENT, description="Check status of the user's order")

test_dispatch_data = [
  [],
  [menu_agent_task],
  [order_agent_task],
  [menu_agent_task, order_agent_task]
]

@pytest.mark.parametrize("tasks", test_dispatch_data)
def test_dispatch_to_agents(tasks: List[AgentTask]):
  state = SnackStackState()
  state[TASKS_FIELD] = tasks
  sends = dispatch_to_agents(state)
  print(sends)
  assert(len(sends) == len(tasks))

  nodes = []
  for send in sends:
    nodes.append(send.node)
    assert send.arg == state

  expected_nodes = []
  for task in tasks:
    expected_nodes.append(task.agent)

  assert nodes == expected_nodes