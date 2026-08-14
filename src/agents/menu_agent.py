"""Defines Menu Agent node for LangGraph"""
from langchain.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agents.state import (
  MENU_AGENT_OUTPUT_FIELD,
  MENU_AGENT_TOOLS,
  GRAPH_END,
  MESSAGES_FIELD,
  REQUIRES_SYNTHESIS_FIELD,
  SYNTHESIZER_AGENT,
  USER_INPUT_FIELD,
  SnackStackState
)
from agents.context_schema import ContextSchema

MENU_INSTRUCTIONS = """
You are a menu agent. Your job is to answer questions about our restaurants menu. 
You can use the *search_menu_catalog* catalog to answer the questions regarding food and menu.
Output rules:
- only answer about menu items you found using the tool
- provide top 3 relevant or related dishes
- if item is not found in the menu say "We do not offer this item at this time."
- if item is not found you can provide some popular items as suggestions to ask about.
- Let user know you specialize in answering questions about the menu and not general internet questions
- You may call the *search_menu_catalog* at most 5 times
"""

def menu_agent_node(state: SnackStackState, runtime: Runtime[ContextSchema]) -> SnackStackState:
  """Menu agent node specializing in answering food and menu related questions"""
  if not state.get(MESSAGES_FIELD):
    messages = [ 
      SystemMessage(content=MENU_INSTRUCTIONS),
      HumanMessage(content=state[USER_INPUT_FIELD])
    ]
  else:
    messages = state[MESSAGES_FIELD]

  result = runtime.context.menu_llm.invoke(messages)
  if result.tool_calls:
    return {MESSAGES_FIELD: [*messages, result] if not state.get(MESSAGES_FIELD) else [result]}

  return {MENU_AGENT_OUTPUT_FIELD: result.content, MESSAGES_FIELD: [result]}

def menu_agent_should_continue(state: SnackStackState) -> str:
  """used to determine if menu_agent needs another turn. Condition on LangGraph edge"""
  if MESSAGES_FIELD in state and len(state[MESSAGES_FIELD]) > 0:
    last_msg = state[MESSAGES_FIELD][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
      return MENU_AGENT_TOOLS
  return SYNTHESIZER_AGENT if state.get(REQUIRES_SYNTHESIS_FIELD) else GRAPH_END
