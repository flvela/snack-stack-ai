"""Defines common constants and functions used for the streamlit app"""
import streamlit as st

# session state config keys
MODEL_CONFIG = "model"
MODEL_PROVIDER_CONFIG = "model_provider"
MODEL_API_KEY_CONFIG = "model_api_key"
EMBEDDINGS_MODEL_PROVIDER_CONFIG = "embeddings_model_provider"
EMBEDDINGS_MODEL_API_KEY_CONFIG = "embeddings_model_api_key"
EMBEDDINGS_MODEL_CONFIG = "embeddings_model"
SNACK_STACK_ASSISTANT_CONFIG = "snack_stack_assistant"
MENU_FILE_CONFIG = "menu_file_config"
ORDERS_FILE_CONFIG = "orders_file_config"
GRAPH_NODE_STATUS_CONFIG = "graph_node_status_config"
REQUIRED_CONFIG = [
  MODEL_CONFIG,
  MODEL_PROVIDER_CONFIG,
  MODEL_API_KEY_CONFIG,
  EMBEDDINGS_MODEL_PROVIDER_CONFIG,
  EMBEDDINGS_MODEL_API_KEY_CONFIG,
  EMBEDDINGS_MODEL_CONFIG,
  MENU_FILE_CONFIG,
  ORDERS_FILE_CONFIG
]
CONFIG_FAILED = "config_failed"

# widget help messages
MENU_FILE_HELP_MESSAGE = """json file expected. Here is an example entry:
  [
    {
      "Dish": "Vegan Pasta Primavera",
      "Cuisine": "Italian",
      "Price": 349,
      "Rating": 4.5,
      "Dietary": "Vegan",
      "Description": "Penne with seasonal vegetables, olive oil, garlic"
    }
  ]
"""
ORDERS_FILE_HELP_MESSAGE = """json file expected. Here is an example entry:
  [
    {
      "Order_ID": "ORD-205",
      "Item": "Paneer Tikka",
      "Customer": "Kavya Sharma",
      "Status": "Placed",
      "Tracking": "SS205TRK",
      "Email": "kavya@example.com"
    }
  ]
"""


def setting_exists(setting_name: str):
  """Helper method to check if setting exists in session state"""
  return (setting_name in st.session_state and st.session_state[setting_name] != ""
          and st.session_state[setting_name] is not None)


def is_config_complete() -> bool:
  """Checks if the user has set all the config required"""
  for setting in REQUIRED_CONFIG:
    if not setting_exists(setting):
      return False
  return True


def config_missing_message() -> str:
  """Gets a list of the missing required config settings"""
  missing_settings = []
  for setting in REQUIRED_CONFIG:
    if not setting_exists(setting):
      missing_settings.append(setting)
  return f"Missing settings are {missing_settings}"
