"""common tools"""
from langchain.tools import tool
from langgraph.types import interrupt


@tool("get_user_input", description="Prompt the user with a message and gets the input from the user.")
def get_user_input(message: str):
  """
  Prompts the user with a message and uses langgraph interrrupt for Human in the loop
  Args:
    message (str): the message to display to the user
    runtime(ToolRuntime): runtime environment for the tool
  """
  print(f"\n###get user input {message}###\n")
  response = interrupt(message)
  return response.strip()


@tool("test_tool", description="tool used for testing purposes.")
def test_tool(query: str):
  """unit testing tool"""
  return query
