import pytest
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime


from agents.context_schema import ContextSchema
from agents.state import MENU_AGENT, MESSAGES_FIELD, ORDER_AGENT, BOTH, ROUTE_FIELD, USER_INPUT_FIELD, SnackStackState
from agents.orchestrator import orchestrator_node
from tools.config import get_llm

context = ContextSchema(orders=None, menu_collection=None, llm=get_llm())
runtime = Runtime(context=context)

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
  assert result[ROUTE_FIELD] == expected_output
  assert len(result[MESSAGES_FIELD]) == 1
  assert result[MESSAGES_FIELD][0].content == user_input

