"""Unit tests for synthesizer agent"""
import pytest

from agents.state import (
  FINAL_RESPONSE_FIELD,
  MENU_AGENT_OUTPUT_FIELD,
  ORDER_AGENT_OUTPUT_FIELD,
  USER_INPUT_FIELD,
  SnackStackState
)
from agents.synthesizer import DEFAULT_SYNTHESIZER_MESSAGE, synthesizer_node
from testutils.common import init_test_runtime

MENU_AGENT_TEST_RESPONSE = "Menu agent test response"
ORDER_AGENT_TEST_RESPONSE = "Order agent test response"
test_data = [
  (None, None, DEFAULT_SYNTHESIZER_MESSAGE),
  (MENU_AGENT_TEST_RESPONSE, None, MENU_AGENT_TEST_RESPONSE),
  (None, ORDER_AGENT_TEST_RESPONSE, ORDER_AGENT_TEST_RESPONSE),
  (MENU_AGENT_TEST_RESPONSE, ORDER_AGENT_TEST_RESPONSE, None)
]

@pytest.mark.parametrize("menu_agent_output, order_agent_output, expected_response", test_data)
def test_synthesizer_node(menu_agent_output:str, order_agent_output:str, expected_response:str):
  """unit test for synthesizer_node"""
  state = SnackStackState()
  state[MENU_AGENT_OUTPUT_FIELD] = menu_agent_output
  state[ORDER_AGENT_OUTPUT_FIELD] = order_agent_output
  state[USER_INPUT_FIELD] = "I would like to order some italian food and check on order 001"
  result = synthesizer_node(state, init_test_runtime())
  print(f"result: {result}")
  if expected_response:
    assert result[FINAL_RESPONSE_FIELD] == expected_response
  else:
    assert result[FINAL_RESPONSE_FIELD] is not None
