"""Synthesizer Agent definition"""
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agents.state import (
  FINAL_RESPONSE_FIELD,
  MENU_AGENT_MESSAGES_FIELD,
  MENU_AGENT_OUTPUT_FIELD,
  MESSAGES_FIELD,
  ORDER_AGENT_MESSAGES_FIELD,
  ORDER_AGENT_OUTPUT_FIELD,
  USER_INPUT_FIELD,
  SnackStackState,
  state_to_string
)
from agents.context_schema import ContextSchema


SYNTHESIZER_INSTRUCTIONS = """
You are a synthesizer agent whose job is to format multiple  agent inputs into a single comprehensive response.
Output rules:
1. Organize information
2. Note any conflicts
3. Do not repeat information
4. End with any actionable next steps.

Original user query:
{query}

Agent results:
{results}
"""
DEFAULT_SYNTHESIZER_MESSAGE = "I could not process your request"


def synthesizer_node(state: SnackStackState, runtime: Runtime[ContextSchema]):
  """defines the synthesizer agent node used by LangGraph"""
  print(f"\n###synthesizer_node###\n{state_to_string(state)}")

  agent_results = []
  if state.get(MENU_AGENT_OUTPUT_FIELD):
    agent_results.append(("Menu agent",
                          "Answering questions on food and menu related items",
                          f"{state.get(MENU_AGENT_OUTPUT_FIELD)}"))

  if state.get(ORDER_AGENT_OUTPUT_FIELD):
    agent_results.append(("Orders Agent",
                          "Answering questions about orders based on order id, email or tracking number",
                          f"{state.get(ORDER_AGENT_OUTPUT_FIELD)}"))

  if len(agent_results) == 1:
    result = agent_results[0][2]
    return {
      FINAL_RESPONSE_FIELD: result,
      MESSAGES_FIELD: [AIMessage(content=result)]
    }

  if len(agent_results) == 0:
    return {
      FINAL_RESPONSE_FIELD: DEFAULT_SYNTHESIZER_MESSAGE
    }

  results_formatted = "\n\n".join([f"""
    {result[0]}
    Focus: {result[1]}
    Result: {result[2]}""" for result in agent_results])

  messages = [
    SystemMessage(content=SYNTHESIZER_INSTRUCTIONS.format(
      query=state.get(USER_INPUT_FIELD), results=results_formatted)),
    HumanMessage(content=state.get(USER_INPUT_FIELD))
  ]

  result = runtime.context.llm.invoke(messages)
  return {FINAL_RESPONSE_FIELD: result.content,
          MESSAGES_FIELD: [result],
          MENU_AGENT_MESSAGES_FIELD: [],
          MENU_AGENT_OUTPUT_FIELD: "",
          ORDER_AGENT_MESSAGES_FIELD: [],
          ORDER_AGENT_OUTPUT_FIELD: ""}
