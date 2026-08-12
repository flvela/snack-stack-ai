
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

@tool("get_user_input", description="Prompt the user with a message and gets the input from the user.")
def get_user_input(message: str, runtime: ToolRuntime):
    """
    Prompts the user with a message and uses langgraph interrrupt for Human in the loop
    Args:
        message (str): the message to display to the user
        runtime(ToolRuntime): runtime environment for the tool
    """
    response = interrupt(message)
    return response.strip()