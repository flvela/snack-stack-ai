"""Defines the SnackStackState, AgentTasks and Orchestrator result"""
from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# graph nodes
MENU_AGENT_NODE = "menu_agent_node"
ORDER_AGENT_NODE = "order_agent_node"
ORCHESTRATOR_AGENT = "orchestrator_agent"
SYNTHESIZER_AGENT = "synthesizer_agent"

# graph states for conditional edges
MENU_AGENT_TOOLS = "menu_agent_tools"
ORDER_AGENT_TOOLS = "order_agent_tools"
GRAPH_END = "end"

# state field names
MESSAGES_FIELD = "messages"
USER_INPUT_FIELD = "user_input"
MENU_AGENT_MESSAGES_FIELD = "menu_agent_messages"
MENU_AGENT_OUTPUT_FIELD = "menu_agent_output"
ORDER_AGENT_MESSAGES_FIELD = "order_agent_messages"
ORDER_AGENT_OUTPUT_FIELD = "order_agent_output"
FINAL_RESPONSE_FIELD = "final_response"
TASKS_FIELD = "tasks"
REQUIRES_SYNTHESIS_FIELD = "requires_synthesis"


class AgentTask(BaseModel):
  """ A task for a specialist agent """
  agent: Literal["menu_agent_node", "order_agent_node"] = Field(
    description="Name of the agent that handles the task"
  )
  description: str = Field(
    description="Description of the task the agent should perform"
  )


class SnackStackState(TypedDict):
  """Class defining the Graph State for the SnackStack assisstant.
  It will be updated by each node in the graph"""
  # conversation history
  messages: Annotated[List[BaseMessage], add_messages]
  # user query
  user_input: str
  # orchestrator decisions
  tasks: List[AgentTask]
  # Flag for synthesis
  requires_synthesis: bool
  # menu agent messages
  menu_agent_messages: Annotated[List[BaseMessage], add_messages]
  # menu agent output
  menu_agent_output: str
  # order agent messages
  order_agent_messages: Annotated[List[BaseMessage], add_messages]
  # order agent output
  order_agent_output: str
  # final response to user
  final_response: str


class OrchestratorResult(BaseModel):
  """The orchestrator agents routing decision"""
  tasks: List[AgentTask] = Field(
    description="Tasks to dispatch"
  )


def state_to_string(state: SnackStackState) -> str:
  """create a formatted string from the snack that state"""
  return "\n".join(f"\n{key}: {value}" for key, value in state.items())
