"""Order Agent Unit tests"""
from langchain.messages import AIMessage, ToolCall
import pytest

from agents.order_agent import order_agent_node, order_agent_should_continue
from agents.state import (
  ORDER_AGENT_MESSAGES_FIELD,
  ORDER_AGENT_OUTPUT_FIELD,
  ORDER_AGENT_TOOLS,
  USER_INPUT_FIELD,
  REQUIRES_SYNTHESIS_FIELD,
  SYNTHESIZER_AGENT,
  SnackStackState
)
from testutils.common import init_test_runtime

GET_USER_INPUT_TOOL = "get_user_input"
SEARCH_ORDER_CATALOG_TOOL = "search_order_catalog"

test_data = [
  #user input and true/false if we should call tools
  ("I want to order some italian food", None),
  #no order number, email or tracking number
  ("what's my order status", GET_USER_INPUT_TOOL),
  #order with email
  ("what's my order status for email test1@gmail.com", SEARCH_ORDER_CATALOG_TOOL),
  #order with tracking id
  ("what's my order status for tracking id 0123", SEARCH_ORDER_CATALOG_TOOL),
  #order with order id
  ("what's my order status for order id 0123", SEARCH_ORDER_CATALOG_TOOL)
]

@pytest.mark.parametrize("user_input, tool_name", test_data)
def test_order_agent_node(user_input: str, tool_name: str):
  """unit test for order_agent_node"""
  state = SnackStackState()
  state[USER_INPUT_FIELD] = user_input

  result = order_agent_node(state, init_test_runtime())
  print(result)
  if tool_name is not None:
    assert len(result[ORDER_AGENT_MESSAGES_FIELD]) > 0
    assert hasattr(result[ORDER_AGENT_MESSAGES_FIELD][-1], "tool_calls")
    tool_calls = result[ORDER_AGENT_MESSAGES_FIELD][-1].tool_calls
    tool_names = [tool_call["name"] for tool_call in tool_calls]
    assert tool_name in tool_names if tool_name else len(tool_calls) == 0
  else:
    assert len(result[ORDER_AGENT_OUTPUT_FIELD]) > 0

tool_call = ToolCall({"name": "foo", "args": {"a": 1}, "id": "123"})


test_state_data =[
  ({}, SYNTHESIZER_AGENT),
  ({ORDER_AGENT_MESSAGES_FIELD: [AIMessage(content="", tool_calls=[tool_call])]},
   ORDER_AGENT_TOOLS),
  ({ORDER_AGENT_MESSAGES_FIELD: []}, SYNTHESIZER_AGENT),
  ({ORDER_AGENT_MESSAGES_FIELD: [{"content", "any_content"}]}, SYNTHESIZER_AGENT),
  ({ORDER_AGENT_MESSAGES_FIELD: [{"content", "any_content"}], REQUIRES_SYNTHESIS_FIELD: True},
   SYNTHESIZER_AGENT),
  ({ORDER_AGENT_MESSAGES_FIELD: [{"content", "any_content"}], REQUIRES_SYNTHESIS_FIELD: False},
   SYNTHESIZER_AGENT)
]

@pytest.mark.parametrize("state, expected_output", test_state_data)
def test_menu_agent_should_continue(state, expected_output):
  """unit test for order_agent_should_continue"""
  result = order_agent_should_continue(state)
  assert expected_output == result
