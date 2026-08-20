"""application page for streamlit app"""
import streamlit as st

from IPython.display import Image

from assistant import SnackStackAssistant
from streamlit_pages.constants import (
  MODEL_PROVIDER_CONFIG,
  MODEL_CONFIG,
  MODEL_API_KEY_CONFIG,
  EMBEDDINGS_MODEL_PROVIDER_CONFIG,
  EMBEDDINGS_MODEL_API_KEY_CONFIG,
  EMBEDDINGS_MODEL_CONFIG,
  SNACK_STACK_ASSISTANT_CONFIG
)
from tools.config import Config
from tools.menu import load_menu_documents
from tools.orders import load_orders_documents

ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"


def setting_exists(setting_name: str):
  """Helper method to check if setting exists in session state"""
  return setting_name in st.session_state and st.session_state[setting_name] != ""


def write_and_save_message(message: str, role: str):
  """stream the response from AI and adds the message to session state"""
  with st.chat_message(role):
    st.write(message)
  st.session_state.messages.append({"role": role, "content": message})


def initialize_assisstant():
  """initializes the AI Assistant"""
  orders = load_orders_documents("data/orders.json")
  st.write(orders)
  menu = load_menu_documents('data/menu.json')
  st.write(menu)
  config = Config(model_provider=st.session_state[MODEL_PROVIDER_CONFIG],
                  model_name=st.session_state[MODEL_CONFIG],
                  model_api_key=st.session_state[MODEL_API_KEY_CONFIG],
                  embeddings_model_provider=st.session_state[EMBEDDINGS_MODEL_PROVIDER_CONFIG],
                  embeddings_model_name=st.session_state[EMBEDDINGS_MODEL_CONFIG],
                  embeddings_model_api_key=st.session_state[EMBEDDINGS_MODEL_API_KEY_CONFIG])
  st.session_state[SNACK_STACK_ASSISTANT_CONFIG] = SnackStackAssistant(orders=orders, menu=menu, config=config)


def is_config_complete() -> bool:
  """Checks if the user has set all the config required"""
  return (setting_exists(MODEL_API_KEY_CONFIG) and setting_exists(MODEL_CONFIG) and setting_exists(MODEL_PROVIDER_CONFIG)
          and setting_exists(EMBEDDINGS_MODEL_CONFIG) and setting_exists(EMBEDDINGS_MODEL_PROVIDER_CONFIG)
          and setting_exists(EMBEDDINGS_MODEL_API_KEY_CONFIG))


def chat_with_user():
  """Implements the user chat"""
  assistant = st.session_state[SNACK_STACK_ASSISTANT_CONFIG]

  if "messages" not in st.session_state:
    st.session_state.messages = []
  # display chat history
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # get new user input
  if prompt := st.chat_input("Say something"):
    write_and_save_message(prompt, USER_ROLE)

    if prompt == "/clear_chat":
      st.session_state.messages = []
      assistant.reset()
    else:
      if not assistant.is_interrupted:
        # start a new chat thread if assistant is not interrupted
        assistant.reset()
      # get response from assistent
      result = assistant.ask(prompt)
      write_and_save_message(result, ASSISTANT_ROLE)


def reset_app():
  """reset the application"""
  st.session_state[SNACK_STACK_ASSISTANT_CONFIG].reset()
  st.session_state.clear()


def ai_assistant_page():
  """defines the streamlit AI Assistant chat page"""
  st.title("Snack Stack AI assistant", text_alignment="left")
  st.divider()
  with st.sidebar:
    st.button("Reset Application", on_click=reset_app)
    st.text_input("Model Provider", help="Enter An AI API Model provider. (ie. anthropic, open_ai, etc)",
                  persist_state="session", key=MODEL_PROVIDER_CONFIG)
    st.text_input("Model", help="Enter the model to used based on Model Provider",
                  persist_state="session", key=MODEL_CONFIG)
    st.text_input("Model API key", type="password", help="Enter the Model API key to use for your model",
                  persist_state="session", key=MODEL_API_KEY_CONFIG)
    st.text_input("Embeddings Model Provider", help="Enter An Embeddings API Model provider. (ie. anthropic, open_ai, etc)",
                  persist_state="session", key=EMBEDDINGS_MODEL_PROVIDER_CONFIG)
    st.text_input("Embeddings Model", help="Enter the embeddings model to used based on Model Provider",
                  persist_state="session", key=EMBEDDINGS_MODEL_CONFIG)
    st.text_input("Embeddings Model API key", type="password", help="Enter the Embeddings Model API key to use for your model",
                  persist_state="session", key=EMBEDDINGS_MODEL_API_KEY_CONFIG)

  if is_config_complete():
    if SNACK_STACK_ASSISTANT_CONFIG not in st.session_state:
      st.warning("SNACK_STACK_ASSISTANT_CONFIG not found")
      initialize_assisstant()

    assistant = st.session_state[SNACK_STACK_ASSISTANT_CONFIG]
    image = Image(data=assistant.graph.get_graph().draw_mermaid_png())
    st.image(image.data, caption="Snack Stack AI langgraph")
    st.info("Successfully configured graph")
    chat_with_user()
  else:
    st.warning("Please configure your model and embeddings provider in the side bar")


ai_assistant_page()
