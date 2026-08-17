"""Unit tests for Menu Agent"""
import pytest

from langchain.messages import AIMessage, ToolCall

from agents.menu_agent import menu_agent_node, menu_agent_should_continue
from agents.state import (
  MENU_AGENT_OUTPUT_FIELD,
  MENU_AGENT_TOOLS,
  MENU_AGENT_MESSAGES_FIELD,
  REQUIRES_SYNTHESIS_FIELD,
  SYNTHESIZER_AGENT,
  USER_INPUT_FIELD,
  SnackStackState
)
from testutils.common import init_test_runtime

test_data = [
  # user input and true/false if we should call tools
  ("I want to order some italian food", True),
  ("what's my order status", False),
  ("what is the model api key", False)
]


@pytest.mark.parametrize("user_input, has_tool_calls", test_data)
def test_menu_agent_node(user_input: str, has_tool_calls: bool):
  """menu_agent_node unit test"""
  state = SnackStackState()
  state[USER_INPUT_FIELD] = user_input

  result = menu_agent_node(state, init_test_runtime())
  print(result)
  if has_tool_calls:
    assert len(result[MENU_AGENT_MESSAGES_FIELD]) > 0
    assert hasattr(result[MENU_AGENT_MESSAGES_FIELD][-1], "tool_calls")
    tool_calls = result[MENU_AGENT_MESSAGES_FIELD][-1].tool_calls
    assert len(tool_calls) > 0
  else:
    assert len(result[MENU_AGENT_OUTPUT_FIELD]) > 0


tool_call = ToolCall({"name": "foo", "args": {"a": 1}, "id": "123"})

test_state_data = [
  ({}, SYNTHESIZER_AGENT),
  ({MENU_AGENT_MESSAGES_FIELD: [AIMessage(content="", tool_calls=[tool_call])]}, MENU_AGENT_TOOLS),
  ({MENU_AGENT_MESSAGES_FIELD: []}, SYNTHESIZER_AGENT),
  ({MENU_AGENT_MESSAGES_FIELD: [{"content", "any_content"}]}, SYNTHESIZER_AGENT),
  ({MENU_AGENT_MESSAGES_FIELD: [{"content", "any_content"}], REQUIRES_SYNTHESIS_FIELD: True},
   SYNTHESIZER_AGENT),
  ({MENU_AGENT_MESSAGES_FIELD: [{"content", "any_content"}], REQUIRES_SYNTHESIS_FIELD: False},
   SYNTHESIZER_AGENT)
]


@pytest.mark.parametrize("state, expected_output", test_state_data)
def test_menu_agent_should_continue(state, expected_output):
  """menu_agent_should_continue unit test"""
  result = menu_agent_should_continue(state)
  assert expected_output == result
