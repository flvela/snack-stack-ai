"""Main page of the streamlit application"""
import json

import streamlit as st
from streamlit_pages.common import (
  CONFIG_FAILED,
  MENU_FILE_CONFIG,
  MENU_FILE_HELP_MESSAGE,
  MODEL_PROVIDER_CONFIG,
  MODEL_CONFIG,
  MODEL_API_KEY_CONFIG,
  EMBEDDINGS_MODEL_PROVIDER_CONFIG,
  EMBEDDINGS_MODEL_API_KEY_CONFIG,
  EMBEDDINGS_MODEL_CONFIG,
  ORDERS_FILE_CONFIG,
  ORDERS_FILE_HELP_MESSAGE,
  SNACK_STACK_ASSISTANT_CONFIG,
  config_missing_message,
  is_config_complete
)

from assistant import SnackStackAssistant
from tools.config import Config
from tools.menu import load_menu_documents_from_json
from tools.orders import load_orders_documents


def initialize_assisstant():
  """initializes the AI Assistant"""
  orders = load_orders_documents('data/orders.json')
  st.write(orders)
  menu = load_menu_documents_from_json(json.load(st.session_state[MENU_FILE_CONFIG]))
  st.write(menu)
  app_config = Config(model_provider=st.session_state[MODEL_PROVIDER_CONFIG],
                      model_name=st.session_state[MODEL_CONFIG],
                      model_api_key=st.session_state[MODEL_API_KEY_CONFIG],
                      embeddings_model_provider=st.session_state[EMBEDDINGS_MODEL_PROVIDER_CONFIG],
                      embeddings_model_name=st.session_state[EMBEDDINGS_MODEL_CONFIG],
                      embeddings_model_api_key=st.session_state[EMBEDDINGS_MODEL_API_KEY_CONFIG])
  st.session_state[SNACK_STACK_ASSISTANT_CONFIG] = SnackStackAssistant(orders=orders, menu=menu, config=app_config)


def configure_application():
  """reset the application"""
  if is_config_complete():
    st.session_state[CONFIG_FAILED] = False
    with st.status("Initializing Assistant") as status:
      initialize_assisstant()
      status.update(label="Assistant Initialized", state="complete")
  else:
    st.session_state[CONFIG_FAILED] = True


assistant_page = st.Page("chat.py", title="Assistant", icon=":material/support_agent:")
menu_page = st.Page("menu.py", title="Menu", icon=":material/menu_book:")
page = st.navigation([assistant_page, menu_page])

config = Config()
with st.sidebar as sidebar:
  st.text_input("Model Provider",
                help="Enter An AI API Model provider. (ie. anthropic, open_ai, etc). The default is in .env file.",
                persist_state="session", key=MODEL_PROVIDER_CONFIG, value=config.model_provider)
  st.text_input("Model",
                help="Enter the model to used based on Model Provider. The default is in .env file.",
                persist_state="session", key=MODEL_CONFIG, value=config.model_name)
  st.text_input("Model API key", type="password",
                help="Enter the Model API key to use for your model. The default is in .env file.",
                persist_state="session", key=MODEL_API_KEY_CONFIG, value=config.model_api_key)
  st.text_input("Embeddings Model Provider",
                help="Enter An Embeddings API Model provider. (ie. anthropic, open_ai, etc). The default is in .env file",
                persist_state="session", key=EMBEDDINGS_MODEL_PROVIDER_CONFIG, value=config.embeddings_model_provider)
  st.text_input("Embeddings Model",
                help="Enter the embeddings model to used based on Model Provider. The default is in .env file",
                persist_state="session", key=EMBEDDINGS_MODEL_CONFIG, value=config.embeddings_model_name)
  st.text_input("Embeddings Model API key", type="password",
                help="Enter the Embeddings Model API key to use for your model. The default is in .env file",
                persist_state="session", key=EMBEDDINGS_MODEL_API_KEY_CONFIG, value=config.embeddings_model_api_key)
  st.file_uploader("Menu File", type="json", help=MENU_FILE_HELP_MESSAGE, key=MENU_FILE_CONFIG,
                   on_change=configure_application)
  st.file_uploader("Orders Files", type="json", help=ORDERS_FILE_HELP_MESSAGE, key=ORDERS_FILE_CONFIG,
                   on_change=configure_application)
  st.button("Configure Application", on_click=configure_application)
  if CONFIG_FAILED in st.session_state and st.session_state[CONFIG_FAILED]:
    st.warning(config_missing_message())


page.run()
