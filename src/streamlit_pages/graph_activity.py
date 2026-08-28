"""Defines Graph activity classes and related functions for display graph status in UI"""
from dataclasses import dataclass
import time


from langgraph.errors import GraphInterrupt

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from streamlit_pages.common import format_time
from agents.state import ORCHESTRATOR_AGENT
from snack_stack_graph import (
  MENU_AGENT_NODE,
  MENU_AGENT_TOOL_NODE,
  ORDER_AGENT_NODE,
  ORDER_AGENT_TOOL_NODE,
  SYNTHESIZER_AGENT_NODE
)

# graph stream event types
EVENT_ON_CHAIN_START = 'on_chain_start'
EVENT_ON_CHAIN_END = "on_chain_end"
EVENT_ON_TOOL_START = "on_tool_start"
EVENT_ON_TOOL_END = "on_tool_end"
EVENT_ON_TOOL_ERROR = "on_tool_error"
EVENT_ON_CHAT_MODEL_END = "on_chat_model_end"
EVENT_ON_PARSER_END = "on_parser_end"

# full graph node
LANGRAPH_NODE_NAME = "LangGraph"

# status texts
STATUS_NOT_STARTED = ":grey-badge[:material/hourglass_top: Not Started]"
STATUS_RUNNING = ":yellow-badge[:material/cached: Running]"
STATUS_PAUSED = ":orange-badge[:material/pause: Paused]"
STATUS_COMPLETED = ":green-badge[:material/check_small: Completed]"

# graph stream event keys
DATA_KEY = 'data'
ERROR_KEY = 'error'
EVENT_KEY = 'event'
FINAL_RESPONSE_KEY = 'final_response'
INPUT_KEY = 'input'
LANGGRAPH_NODE_KEY = 'langgraph_node'
MENU_AGENT_OUTPUT_KEY = 'menu_agent_output'
METADATA_KEY = "metadata"
NAME_KEY = 'name'
ORDER_AGENT_OUTPUT_KEY = 'order_agent_output'
OUTPUT_KEY = 'output'
TASKS_KEY = 'tasks'
THREAD_ID_KEY = 'thread_id'
USER_INPUT_KEY = 'user_input'


@dataclass
class GraphNodeStatus:
  """used to keep track and display of graph node status """
  # name of the graph node as it it appear in langgraph
  name: str
  # UI status placeholder to write node status updates
  status_placeholder: DeltaGenerator
  # start time for the current run
  start_time: float
  # UI time placeholder to write node status execution time
  time_placeholder: DeltaGenerator
  # total run time for node across multiple runs
  total_run_time: float
  # the status text to display in UI
  status_text: str

  def update_status(self, state: str):
    """updates the status of the node"""
    if self.status_placeholder:
      self.status_placeholder.markdown(state, text_alignment="right")
    self.status_text = state

    if state == STATUS_RUNNING:
      self.start_time = time.perf_counter()
    elif state == STATUS_COMPLETED:
      end_time = time.perf_counter()
      elapsed_time_ms = (end_time - self.start_time) * 1000
      self.total_run_time = self.total_run_time + elapsed_time_ms
      if self.time_placeholder:
        self.time_placeholder.markdown(f"{format_time(self.total_run_time)}", text_alignment="right")


def build_graph_state_placeholder(label: str, state_parent: DeltaGenerator) -> DeltaGenerator:
  """builds a graph state placeholder in the state parent container with the given label"""
  label_container = state_parent.container(border=True, gap="xxsmall")
  label_col, data_col = label_container.columns([1, 4])
  label_col.write(label)
  placeholder = data_col.empty()
  placeholder.markdown("--", text_alignment="left")
  return placeholder


@dataclass
class GraphStateActivity:
  """Keep track of UI placeholders to display graph state"""
  # label to placeholder mapping
  label_to_placeholder_mapping: dict
  # labels that are optional
  optional_data_labels: set

  def __init__(self, state_parent: DeltaGenerator):
    """creates all the containers and placeholders in the state parent container for UI to display graph state"""
    self.label_to_placeholder_mapping = {
      THREAD_ID_KEY: build_graph_state_placeholder(THREAD_ID_KEY, state_parent),
      USER_INPUT_KEY: build_graph_state_placeholder(USER_INPUT_KEY, state_parent),
      TASKS_KEY: build_graph_state_placeholder(TASKS_KEY, state_parent),
      MENU_AGENT_OUTPUT_KEY: build_graph_state_placeholder(MENU_AGENT_OUTPUT_KEY, state_parent),
      ORDER_AGENT_OUTPUT_KEY: build_graph_state_placeholder(ORDER_AGENT_OUTPUT_KEY, state_parent),
      FINAL_RESPONSE_KEY: build_graph_state_placeholder(FINAL_RESPONSE_KEY, state_parent),
    }
    self.optional_data_labels = set([USER_INPUT_KEY, TASKS_KEY, MENU_AGENT_OUTPUT_KEY, ORDER_AGENT_OUTPUT_KEY,
                                    FINAL_RESPONSE_KEY])

  def update_state(self, langgraph_event_item):
    """updates the graph status UI based on the langraph event"""
    self.label_to_placeholder_mapping[THREAD_ID_KEY].markdown(
      langgraph_event_item[METADATA_KEY][THREAD_ID_KEY], text_alignment="left")
    if DATA_KEY in langgraph_event_item:
      event_data = langgraph_event_item[DATA_KEY]
      if INPUT_KEY in event_data:
        input_data = event_data[INPUT_KEY]
        if isinstance(input_data, dict) and USER_INPUT_KEY in input_data:
          self.label_to_placeholder_mapping[USER_INPUT_KEY].markdown(input_data[USER_INPUT_KEY], text_alignment="left")
      if OUTPUT_KEY in event_data:
        output_data = event_data[OUTPUT_KEY]
        for label in self.optional_data_labels:
          if label in output_data:
            self.label_to_placeholder_mapping[label].markdown(output_data[label], text_alignment="left")


@dataclass
class GraphEventActivity:
  """class used to display events streamed from Langgraph"""
  parent_event_containers: dict
  child_event_containers: dict[dict]
  events_by_node_name: dict[str, list]
  events_container: DeltaGenerator

  def __init__(self, events_container: DeltaGenerator):
    self.parent_event_containers = {}
    self.child_event_containers = {}
    self.events_by_node_name = {}
    self.events_container = events_container

  def update_event(self, langgraph_item, node_status: GraphNodeStatus):
    """Updates the UI event container based on the given langgraph event item"""

    node_name = langgraph_item[NAME_KEY]
    if LANGGRAPH_NODE_KEY in langgraph_item[METADATA_KEY]:
      parent_name = langgraph_item[METADATA_KEY][LANGGRAPH_NODE_KEY]
    else:
      parent_name = node_name

    if parent_name not in self.parent_event_containers:
      parent_container = self.events_container.status(parent_name, expanded=True)
      self.parent_event_containers[parent_name] = parent_container
      self.child_event_containers[parent_name] = {}
    else:
      parent_container = self.parent_event_containers[parent_name]

    if parent_name != node_name and node_name not in self.child_event_containers[parent_name]:
      self.child_event_containers[parent_name][node_name] = parent_container.status(node_name, expanded=True)

    event_key = langgraph_item[EVENT_KEY]
    state_complete = event_key in [EVENT_ON_CHAIN_END, EVENT_ON_CHAT_MODEL_END, EVENT_ON_TOOL_END, EVENT_ON_PARSER_END]
    state_text = "complete" if state_complete else "running"
    if parent_name != node_name:
      status_container = self.child_event_containers[parent_name][node_name]
      with status_container.expander(event_key):
        st.write(langgraph_item)
      status_container.update(label=node_name, state=state_text, expanded=not state_complete)
    else:
      status_container = self.parent_event_containers[parent_name]
      with status_container.expander(event_key):
        st.write(langgraph_item)
      status_container.update(label=parent_name, state=state_text, expanded=not state_complete)
      if state_complete:
        del self.parent_event_containers[parent_name]


@dataclass
class GraphActivity:
  """class used to show Langraph node status in the UI"""
  node_statuses_by_name: dict
  state_activity: GraphStateActivity
  events_activity: GraphEventActivity

  def __init__(self):
    self.node_statuses_by_name = {}
    self.state_activity = None
    self.events_activity = None

  def initialize_state_container(self, state_container: DeltaGenerator):
    """initializes the state activity with the UI state container"""
    self.state_activity = GraphStateActivity(state_container)

  def initialize_events_container(self, events_container: DeltaGenerator):
    """initializes the events activity with the UI events container"""
    self.events_activity = GraphEventActivity(events_container)

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

  def initialize_graph_node(self, node_name: str, parent_container: DeltaGenerator):
    """initializes the graph status and placeholder status container in the parent_container"""
    node_container = parent_container.container(key=node_name, border=True, gap="xxsmall")
    node_col, status_col = node_container.columns(2)
    node_col.write(node_name.replace("_", " ").title())
    status_placeholder = status_col.empty()
    time_placeholder = status_col.empty()
    if node_name in self.node_statuses_by_name:
      graph_node = self.node_statuses_by_name[node_name]
      graph_node.status_placeholder = status_placeholder
      graph_node.time_placeholder = time_placeholder
    else:
      graph_node = GraphNodeStatus(
        node_name,
        status_placeholder=status_placeholder,
        start_time=0,
        time_placeholder=time_placeholder,
        total_run_time=0,
        status_text=STATUS_NOT_STARTED)
      self.node_statuses_by_name[node_name] = graph_node

    node_status = graph_node.status_text
    time_text = format_time(graph_node.total_run_time)
    status_placeholder.markdown(node_status, text_alignment="right")
    time_placeholder.markdown(time_text, text_alignment="right")

  def update_node_status(self, langgraph_event_item):
    """updates the node status in UI based on the node name and langgraph_event_item"""
    node_name = langgraph_event_item[NAME_KEY]
    if node_name in self.node_statuses_by_name:
      # if node name exists it means it is tracked in graph tab
      node_status = self.node_statuses_by_name[node_name]
    else:
      # for nodes that are not tracked in the graph tab. keep the status for timing
      node_status = GraphNodeStatus(
        name=node_name,
        status_placeholder=None,
        time_placeholder=None,
        total_run_time=0,
        start_time=0,
        status_text="--"
      )
      self.node_statuses_by_name[node_name] = node_status

    if langgraph_event_item[EVENT_KEY] in (EVENT_ON_CHAIN_START, EVENT_ON_TOOL_START):
      node_status.update_status(STATUS_RUNNING)
      return node_status

    if langgraph_event_item[EVENT_KEY] in (EVENT_ON_CHAIN_END):
      node_status.update_status(STATUS_COMPLETED)
      return node_status

    if langgraph_event_item[EVENT_KEY] in EVENT_ON_TOOL_ERROR:
      node_tool_name = langgraph_event_item[METADATA_KEY][LANGGRAPH_NODE_KEY]
      error = langgraph_event_item[DATA_KEY][ERROR_KEY]
      if isinstance(error, GraphInterrupt) and node_tool_name in self.node_statuses_by_name:
        self.node_statuses_by_name[node_tool_name].update_status(STATUS_PAUSED)

    return node_status

  def update(self, langgraph_event_item):
    """Main method to update Graph activity for Graph, State and Events tab based on LangGraph events"""
    if self.state_activity is not None:
      self.state_activity.update_state(langgraph_event_item)
    node_status = self.update_node_status(langgraph_event_item)
    if self.events_activity is not None:
      self.events_activity.update_event(langgraph_event_item, node_status)

  def reset_node_statuses(self):
    """reset the graph node order status and time placeholders"""
    for _, node in self.node_statuses_by_name.items():
      node.status_text = STATUS_NOT_STARTED
      node.total_run_time = 0
      node.start_time = 0
      if node.status_placeholder:
        node.status_placeholder.markdown(STATUS_NOT_STARTED, text_alignment="right")
      if node.time_placeholder:
        node.time_placeholder.markdown("--", text_alignment="right")
