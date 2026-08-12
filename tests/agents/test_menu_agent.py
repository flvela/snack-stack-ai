from agents.menu_agent import menu_agent_node, menu_agent_should_continue
from agents.state import GRAPH_END, MENU_AGENT_TOOLS, MESSAGES_FIELD, REQUIRES_SYNTHESIS_FIELD, SYNTHESIZER_AGENT, USER_INPUT_FIELD, SnackStackState
from langchain.messages import AIMessage, ToolCall
import pytest

from agents.context_schema import ContextSchema
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime
from tools.config import get_llm

test_data = [
  #user input and true/false if we should call tools
  ("I want to order some italian food", True),
  ("what's my order status", False),
  ("what is the model api key", False)
]

context = ContextSchema(orders=None, menu_collection=None, llm=get_llm())
runtime = Runtime(context=context)

@pytest.mark.parametrize("user_input, has_tool_calls", test_data)
def test_menu_agent_node(user_input: str, has_tool_calls: bool):
  state = SnackStackState()
  state[USER_INPUT_FIELD] = user_input

  result = menu_agent_node(state, runtime)
  print(result)
  assert(len(result[MESSAGES_FIELD])) > 0
  assert hasattr(result[MESSAGES_FIELD][-1], "tool_calls") 
  tool_calls = result[MESSAGES_FIELD][-1].tool_calls
  assert  len(tool_calls) > 0 if has_tool_calls else len(tool_calls) == 0
  

tool_call = ToolCall({"name": "foo", "args": {"a": 1}, "id": "123"})


test_state_data =[
  ({}, GRAPH_END),
  ({MESSAGES_FIELD: [AIMessage(content="", tool_calls=[tool_call])]}, MENU_AGENT_TOOLS),
  ({MESSAGES_FIELD: []}, GRAPH_END),
  ({MESSAGES_FIELD: [{"content", "any_content"}]}, GRAPH_END),
  ({MESSAGES_FIELD: [{"content", "any_content"}], REQUIRES_SYNTHESIS_FIELD: True}, SYNTHESIZER_AGENT),
  ({MESSAGES_FIELD: [{"content", "any_content"}], REQUIRES_SYNTHESIS_FIELD: False}, GRAPH_END)
]

@pytest.mark.parametrize("state, expected_output", test_state_data)
def test_menu_agent_should_continue(state, expected_output):
  result = menu_agent_should_continue(state)
  assert expected_output == result