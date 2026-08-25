"""application page for streamlit app"""
import asyncio
from dataclasses import dataclass
import time

from langgraph.errors import GraphInterrupt

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_pages.common import (
  GRAPH_NODE_STATUS_CONFIG,
  SNACK_STACK_ASSISTANT_CONFIG,
  config_missing_message,
  is_config_complete
)

from agents.state import ORCHESTRATOR_AGENT
from snack_stack_graph import (
  MENU_AGENT_NODE,
  MENU_AGENT_TOOL_NODE,
  ORDER_AGENT_NODE,
  ORDER_AGENT_TOOL_NODE,
  SYNTHESIZER_AGENT_NODE
)

# chat roles
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"

# graph stream event keys
DATA_KEY = 'data'
ERROR_KEY = 'error'
EVENT_KEY = 'event'
LANGGRAPH_NODE_KEY = 'langgraph_node'
METADATA_KEY = 'metadata'
NAME_KEY = 'name'

# graph stream event types
EVENT_ON_CHAIN_START = 'on_chain_start'
EVENT_ON_CHAIN_END = "on_chain_end"
EVENT_ON_TOOL_START = "on_tool_start"
EVENT_ON_TOOL_END = "on_tool_end"
EVENT_ON_TOOL_ERROR = "on_tool_error"

# full graph node
LANGRAPH_NODE_NAME = "LangGraph"

# status texts
STATUS_NOT_STARTED = ":grey-badge[:material/hourglass_top: Not Started]"
STATUS_RUNNING = ":yellow-badge[:material/cached: Running]"
STATUS_PAUSED = ":orange-badge[:material/pause: Paused]"
STATUS_COMPLETED = ":green-badge[:material/check_small: Completed]"


def write_and_save_message(message: str, role: str, message_container):
  """stream the response from AI and adds the message to session state"""
  with message_container:
    with st.chat_message(role):
      st.write(message)
    st.session_state.messages.append({"role": role, "content": message})


def format_time(time_ms: float) -> str:
  """returns a string of the time + units.
    Ex 1: time_ms = 1000 -> 1 s
    Ex 2: time_ms = 10 -> 10 ms
  """
  if time_ms == 0:
    return "--"
  return f"{time_ms/1000:.2f} s" if time_ms > 900 else f"{time_ms:.2f} ms"


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


async def graph_stream_events(prompt: str, events_container):
  """Streams graph event status to the UI"""
  assistant = st.session_state[SNACK_STACK_ASSISTANT_CONFIG]
  node_statuses_by_name = st.session_state[GRAPH_NODE_STATUS_CONFIG]

  stream = await assistant.async_ask(prompt)

  async for item in stream:
    node_name = item[NAME_KEY]
    events_container.write(item)
    if item[EVENT_KEY] in (EVENT_ON_CHAIN_START, EVENT_ON_TOOL_START):
      if node_name in node_statuses_by_name:
        node_status = node_statuses_by_name[node_name]
        node_status.start_time = time.perf_counter()
        node_status.status_placeholder.markdown(STATUS_RUNNING, text_alignment="right")
        node_status.status_text = STATUS_RUNNING
    elif item[EVENT_KEY] in (EVENT_ON_CHAIN_END):
      if node_name in node_statuses_by_name:
        node_status = node_statuses_by_name[node_name]
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - node_status.start_time) * 1000
        node_status.total_run_time = node_status.total_run_time + elapsed_time_ms
        node_status.status_placeholder.markdown(STATUS_COMPLETED, text_alignment="right")
        node_status.time_placeholder.markdown(f"{format_time(node_status.total_run_time)}", text_alignment="right")
        node_status.status_text = STATUS_COMPLETED
    elif item[EVENT_KEY] in EVENT_ON_TOOL_ERROR:
      node_tool_name = item[METADATA_KEY][LANGGRAPH_NODE_KEY]
      if node_tool_name in node_statuses_by_name:
        error = item[DATA_KEY][ERROR_KEY]
        if isinstance(error, GraphInterrupt):
          node_statuses_by_name[node_tool_name].status_placeholder.markdown(STATUS_PAUSED, text_alignment="right")
          node_statuses_by_name[node_tool_name].status_text = STATUS_PAUSED


def reset_graph_node(node_name: str):
  """reset the node status and times"""
  if node_name in st.session_state[GRAPH_NODE_STATUS_CONFIG]:
    node = st.session_state[GRAPH_NODE_STATUS_CONFIG][node_name]
    node.status_text = STATUS_NOT_STARTED
    node.total_run_time = 0
    node.start_time = 0
    node.status_placeholder.markdown(STATUS_NOT_STARTED, text_alignment="right")
    node.time_placeholder.markdown("--", text_alignment="right")


def reset_node_statuses():
  """reset the graph node order status and time placeholders"""
  reset_graph_node(ORCHESTRATOR_AGENT)
  reset_graph_node(MENU_AGENT_NODE)
  reset_graph_node(ORDER_AGENT_NODE)
  reset_graph_node(MENU_AGENT_TOOL_NODE)
  reset_graph_node(ORDER_AGENT_TOOL_NODE)
  reset_graph_node(SYNTHESIZER_AGENT_NODE)


def async_chat_with_user(message_container, events_container):
  """asynchronous chat with user.
  Args:
    messsage_container: the message container for chat history
    node_statuses_by_name (dict): graph node container to write messages to as graph streams events
  """
  assistant = st.session_state[SNACK_STACK_ASSISTANT_CONFIG]

  if "messages" not in st.session_state:
    st.session_state.messages = []
  # display chat history
  with message_container:
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  # get new user input
  if prompt := st.chat_input("Say something"):
    write_and_save_message(prompt, USER_ROLE, message_container)

    if prompt == "/clear_chat":
      st.session_state.messages = []
      assistant.reset()
    else:
      if assistant.get_interrupt_value() is None:
        # start a new chat thread if assistant is not interrupted
        assistant.reset()
        reset_node_statuses()
      # stream events from assistant
      with message_container.status(label="Processing", state="running"):
        asyncio.run(graph_stream_events(prompt, events_container))
      result = assistant.get_last_message()
      write_and_save_message(result, ASSISTANT_ROLE, message_container)


def initialize_graph_node(node_name: str, parent_container: DeltaGenerator):
  """initializes the graph status and placeholder status container in the parent_container"""
  node_container = parent_container.container(key=node_name, border=True, gap="xxsmall")
  node_col, status_col = node_container.columns(2)
  node_col.write(node_name.replace("_", " ").title())
  status_placeholder = status_col.empty()
  time_placeholder = status_col.empty()
  if node_name in st.session_state[GRAPH_NODE_STATUS_CONFIG]:
    graph_node = st.session_state[GRAPH_NODE_STATUS_CONFIG][node_name]
    graph_node.status_placeholder = status_placeholder
    graph_node.time_placeholder = time_placeholder
  else:
    graph_node = GraphNodeStatus(
      ORCHESTRATOR_AGENT,
      status_placeholder=status_placeholder,
      start_time=0,
      time_placeholder=time_placeholder,
      total_run_time=0,
      status_text=STATUS_NOT_STARTED)
    st.session_state[GRAPH_NODE_STATUS_CONFIG][node_name] = graph_node

  node_status = graph_node.status_text
  time_text = format_time(graph_node.total_run_time)
  status_placeholder.markdown(node_status, text_alignment="right")
  time_placeholder.markdown(time_text, text_alignment="right")


def build_graph_containers(graph_container):
  """Builds the UI containers to show graph status"""
  initialize_graph_node(ORCHESTRATOR_AGENT, graph_container)
  menu_and_orders_container = graph_container.container(border=False, gap="xxsmall")
  col1, col2 = menu_and_orders_container.columns(2)
  initialize_graph_node(MENU_AGENT_NODE, col1)
  initialize_graph_node(ORDER_AGENT_NODE, col2)
  tools_and_synthesizer_container = graph_container.container(border=False, gap='xxsmall')
  col1, col2 = tools_and_synthesizer_container.columns(2)
  initialize_graph_node(MENU_AGENT_TOOL_NODE, col1)
  initialize_graph_node(SYNTHESIZER_AGENT_NODE, graph_container)
  initialize_graph_node(ORDER_AGENT_TOOL_NODE, col2)


def async_ai_assistant_page():
  """defines the streamlit AI Assistant chat page"""
  st.set_page_config(layout="wide")

  st.header("Snack Stack AI Assistant", text_alignment="left")
  st.divider()
  if GRAPH_NODE_STATUS_CONFIG not in st.session_state:
    st.session_state[GRAPH_NODE_STATUS_CONFIG] = {}

  if is_config_complete() and SNACK_STACK_ASSISTANT_CONFIG in st.session_state:
    col1, col2 = st.columns(2)
    col1.subheader("Conversation")
    col2.subheader("Graph Activity")
    col2.badge(label="Graph activity populated after first query", color="blue", icon=":material/info:")

    graph_tab, events_tab = col2.tabs(["Graph", "Events"])
    events_container = events_tab.container(key="events_container", height=500, border=True, gap="xxsmall")
    with col2:
      graph_container = graph_tab.container(key="graph_container", height=500, border=False, gap="xxsmall", width="content")
      build_graph_containers(graph_container)
    with col1:
      message_container = st.container(key="message_container", height=480, border=False, gap="xxsmall")
      async_chat_with_user(message_container, events_container)

    st.write(st.session_state)
  else:
    st.warning(f"Please configure your application in sidebar. {config_missing_message()}")


async_ai_assistant_page()
