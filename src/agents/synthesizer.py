from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agents.state import FINAL_RESPONSE_FIELD, MENU_AGENT_OUTPUT_FIELD, MESSAGES_FIELD, ORDER_AGENT_OUTPUT_FIELD, USER_INPUT_FIELD, SnackStackState
from agents.context_schema import ContextSchema


synthesizer_instructions="""
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
def synthesizer_node(state: SnackStackState, runtime:Runtime[ContextSchema]):
  if not state.get(ORDER_AGENT_OUTPUT_FIELD) and not state.get(MENU_AGENT_OUTPUT_FIELD):
    
    return {FINAL_RESPONSE_FIELD: DEFAULT_SYNTHESIZER_MESSAGE,
            MESSAGES_FIELD: AIMessage(content=DEFAULT_SYNTHESIZER_MESSAGE)}

  if state.get(ORDER_AGENT_OUTPUT_FIELD) and not state.get(MENU_AGENT_OUTPUT_FIELD):
    return {FINAL_RESPONSE_FIELD: state.get(ORDER_AGENT_OUTPUT_FIELD),
            MESSAGES_FIELD: AIMessage(content=state.get(ORDER_AGENT_OUTPUT_FIELD))}

  if state.get(MENU_AGENT_OUTPUT_FIELD) and not state.get(ORDER_AGENT_OUTPUT_FIELD):
    return {FINAL_RESPONSE_FIELD: state.get(MENU_AGENT_OUTPUT_FIELD),
            MESSAGES_FIELD: AIMessage(content=state.get(MENU_AGENT_OUTPUT_FIELD))}

  results_formatted = f"""
      Menu agent
      Focus: Answering questions on food and menu related items
      Result: {state.get(MENU_AGENT_OUTPUT_FIELD)}

      Orders Agent
      Focus: Answering questions about orders based on order id, email or tracking number
      Result: {state.get(ORDER_AGENT_OUTPUT_FIELD)}
  """

  messages = [
    SystemMessage(content=synthesizer_instructions.format(query=state.get(USER_INPUT_FIELD), results=results_formatted)),
    HumanMessage(content=state.get(USER_INPUT_FIELD))
  ]

  result = runtime.context.llm.invoke(messages)
  return {FINAL_RESPONSE_FIELD: result.content,
          MESSAGES_FIELD: [result]}
  