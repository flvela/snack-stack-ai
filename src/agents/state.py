from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

#graph nodes
MENU_AGENT = "menu_agent"
ORDER_AGENT = "order_agent"
ORCHESTRATOR_AGENT = "orchestrator_agent"
SYNTHESIZER_AGENT = "synthesizer_agent"

#graph states for conditional edges
MENU_AGENT_TOOLS = "menu_agent_tools"
ORDER_AGENT_TOOLS = "order_agent_tools"
GRAPH_END = "end"

#state field names
MESSAGES_FIELD = "messages"
USER_INPUT_FIELD = "user_input"
MENU_AGENT_OUTPUT_FIELD = "menu_agent_output"
ORDER_AGENT_OUTPUT_FIELD = "order_agent_output"
FINAL_RESPONSE_FIELD = "final_response"
TASKS_FIELD = "tasks"
REQUIRES_SYNTHESIS_FIELD="requires_synthesis"

class SnackStackState(TypedDict):
  """ Class defining the Graph State for the SnackStack assisstant. It will be updated by each node in the graph"""
  #conversation history
  messages: Annotated[List[BaseMessage], add_messages]
  #user query
  user_input: str
  #orchestrator decisions
  tasks: List[AgentTask]
  #Flag for synthesis
  requires_synthesis: bool
  #menu agent output
  menu_agent_output: str
  #order agent output
  order_agent_output: str
  #final response to user
  final_response: str

class AgentTask(BaseModel):
  """ A task for a specialist agent """
  agent: Literal["menu_agent", "order_agent"] = Field(
    description="Name of the agent that handles the task"
  )
  description: str = Field(
    description="Description of the task the agent should perform"
  )

class OrchestratorResult(BaseModel):
  """The orchestrator agents routing decision"""
  tasks: List[AgentTask] = Field(
    description="Tasks to dispatch"
  )