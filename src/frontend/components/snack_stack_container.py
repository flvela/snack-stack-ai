"""SnackStack AI Activity container implementation"""
from streamlit.delta_generator import DeltaGenerator

from agents.state import (
  FINAL_RESPONSE_FIELD,
  MENU_AGENT_OUTPUT_FIELD,
  ORCHESTRATOR_AGENT,
  ORDER_AGENT_OUTPUT_FIELD,
  TASKS_FIELD,
  USER_INPUT_FIELD
)
from snack_stack_graph import (
  MENU_AGENT_NODE,
  MENU_AGENT_TOOL_NODE,
  ORDER_AGENT_NODE,
  ORDER_AGENT_TOOL_NODE,
  SYNTHESIZER_AGENT_NODE
)
from frontend.components.graph_containers import (
  GraphActivityContainer,
  LANGRAPH_NODE_NAME,
  build_graph_state_placeholder
)


class SnackStackActivityContainer(GraphActivityContainer):
  """SnackStackAI graph container class used to display graph nodes, state and events in UI"""

  def initialize_nodes_container(self, graph_container: DeltaGenerator):
    """Builds the UI containers to show graph status in the graph_container"""
    self.initialize_graph_node(ORCHESTRATOR_AGENT, graph_container)
    menu_and_orders_container = graph_container.container(border=False, gap="xxsmall")
    col1, col2 = menu_and_orders_container.columns(2)
    self.initialize_graph_node(MENU_AGENT_NODE, col1)
    self.initialize_graph_node(ORDER_AGENT_NODE, col2)
    tools_and_synthesizer_container = graph_container.container(border=False, gap='xxsmall')
    col1, col2 = tools_and_synthesizer_container.columns(2)
    self.initialize_graph_node(MENU_AGENT_TOOL_NODE, col1)
    self.initialize_graph_node(SYNTHESIZER_AGENT_NODE, graph_container)
    self.initialize_graph_node(ORDER_AGENT_TOOL_NODE, col2)
    self.initialize_graph_node(LANGRAPH_NODE_NAME, graph_container)

  def get_state_field_name_placeholders(self, state_container: DeltaGenerator):
    """initialized the field name to placeholder container to display graph state in UI"""
    return {
      USER_INPUT_FIELD: build_graph_state_placeholder(USER_INPUT_FIELD, state_container),
      TASKS_FIELD: build_graph_state_placeholder(TASKS_FIELD, state_container),
      MENU_AGENT_OUTPUT_FIELD: build_graph_state_placeholder(MENU_AGENT_OUTPUT_FIELD, state_container),
      ORDER_AGENT_OUTPUT_FIELD: build_graph_state_placeholder(ORDER_AGENT_OUTPUT_FIELD, state_container),
      FINAL_RESPONSE_FIELD: build_graph_state_placeholder(FINAL_RESPONSE_FIELD, state_container),
    }
