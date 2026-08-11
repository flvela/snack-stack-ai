import pytest
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime


from agents.context_schema import ContextSchema
from agents.state import FINAL_RESPONSE_FIELD, MENU_AGENT_OUTPUT_FIELD, MESSAGES_FIELD, ORDER_AGENT_OUTPUT_FIELD, USER_INPUT_FIELD, SnackStackState
from agents.synthesizer import DEFAULT_SYNTHESIZER_MESSAGE, synthesizer_node
from tools.config import get_llm

context = ContextSchema(orders=None, menu_collection=None, llm=get_llm())
runtime = Runtime(context=context)

menu_agent_test_response = "Menu agent test response"
order_agent_test_response = "Order agent test response"
test_data = [
  (None, None, DEFAULT_SYNTHESIZER_MESSAGE),
  (menu_agent_test_response, None, menu_agent_test_response),
  (None, order_agent_test_response, order_agent_test_response),
  (menu_agent_test_response, order_agent_test_response, None)
]

@pytest.mark.parametrize("menu_agent_output, order_agent_output, expected_response", test_data)
def test_synthesizer_node(menu_agent_output:str, order_agent_output:str, expected_response:str):
  state = SnackStackState()
  state[MENU_AGENT_OUTPUT_FIELD] = menu_agent_output
  state[ORDER_AGENT_OUTPUT_FIELD] = order_agent_output
  state[USER_INPUT_FIELD] = "I would like to order some italian food and check on order 001"
  result = synthesizer_node(state, runtime)
  if expected_response:
    assert result[FINAL_RESPONSE_FIELD] == expected_response  
  else:
    assert result[FINAL_RESPONSE_FIELD] != None
  