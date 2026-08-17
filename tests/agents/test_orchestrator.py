"""Unit tests for orchestrator agent"""
from typing import List

from langchain.messages import HumanMessage
import pytest

from agents.state import (
  FINAL_RESPONSE_FIELD,
  MENU_AGENT_MESSAGES_FIELD,
  MENU_AGENT_NODE,
  MENU_AGENT_OUTPUT_FIELD,
  ORDER_AGENT_MESSAGES_FIELD,
  ORDER_AGENT_NODE,
  ORDER_AGENT_OUTPUT_FIELD,
  USER_INPUT_FIELD,
  TASKS_FIELD,
  REQUIRES_SYNTHESIS_FIELD,
  AgentTask,
  SnackStackState
)
from agents.orchestrator import orchestrator_node, dispatch_to_agents
from testutils.common import init_test_runtime

BOTH = "both"

test_data = [
  ("Can you recommend some menu dishes to me", MENU_AGENT_NODE),
  ("What's the status of my order", ORDER_AGENT_NODE),
  ("Tell me about the reviews for Chicken tikka masala and my order status", BOTH),
  ("How's the weather in California", MENU_AGENT_NODE)
]


@pytest.mark.parametrize("user_input, expected_output", test_data)
def test_orchestrator_node(user_input: str, expected_output: str):
  """tests the orchestrator node"""
  state = SnackStackState()
  state[USER_INPUT_FIELD] = user_input
  result = orchestrator_node(state, init_test_runtime())
  print(result)
  tasks = result[TASKS_FIELD]

  agents = []
  for task in tasks:
    agents.append(task.agent)

  if expected_output == BOTH:
    assert len(agents) == 2
    assert MENU_AGENT_NODE in agents
    assert MENU_AGENT_NODE in agents
    assert result[REQUIRES_SYNTHESIS_FIELD]
  else:
    assert len(agents) == 1
    assert expected_output in agents
    assert not result[REQUIRES_SYNTHESIS_FIELD]

  assert result[USER_INPUT_FIELD] == user_input


menu_agent_task = AgentTask(agent=MENU_AGENT_NODE,
                             description="Can you recommend some dishes to me")
order_agent_task = AgentTask(agent=ORDER_AGENT_NODE,
                              description="Check status of the user's order")

test_dispatch_data = [
  [],
  [menu_agent_task],
  [order_agent_task],
  [menu_agent_task, order_agent_task]
]


@pytest.mark.parametrize("tasks", test_dispatch_data)
def test_dispatch_to_agents(tasks: List[AgentTask]):
  """unit test for dispatch to agents"""
  state = SnackStackState()
  state[TASKS_FIELD] = tasks
  state[ORDER_AGENT_MESSAGES_FIELD] = [HumanMessage(content="hello")]
  sends = dispatch_to_agents(state)
  print(sends)
  assert len(sends) == len(tasks)

  expected_worker_state = {
    **state,
    MENU_AGENT_MESSAGES_FIELD: [],
    MENU_AGENT_OUTPUT_FIELD: "",
    ORDER_AGENT_MESSAGES_FIELD: [],
    ORDER_AGENT_OUTPUT_FIELD: "",
    FINAL_RESPONSE_FIELD: "",
  }
  nodes = []
  for send in sends:
    nodes.append(send.node)
    assert send.arg == expected_worker_state

  expected_nodes = []
  for task in tasks:
    expected_nodes.append(task.agent)

  assert nodes == expected_nodes
