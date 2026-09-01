"""Order Agent tools"""
import json
from multi_key_dict import multi_key_dict
from langchain.tools import ToolRuntime, tool


# ORDERS CONSTANTS
ORDER_ID = "Order_ID"
ORDER_ITEM = "Item"
ORDER_CUSTOMER = "Customer"
ORDER_STATUS = "Status"
ORDER_TRACKING = "Tracking"
ORDER_EMAIL = "Email"

ORDERS_NAME = "orders"


def load_orders_documents(file_path: str) -> multi_key_dict:
  """
  Load order documents from a JSON file.

  Args:
    file_path (str): Path to the JSON file containing order documents.
  Returns
    multi_key_dict: A multi-key dictionary of order documents,
                    keyed by Order_ID, Order_Tracking, and Order_Email.
  """
  with open(file_path, 'r', encoding='utf-8') as file:
    orders = json.load(file)

  return load_orders_documents_from_json(orders)


def load_orders_documents_from_json(json_documents: list[dict]) -> multi_key_dict:
  """
  Load order documents from a JSON list.

  Args:
    json_documents (str): json list of orders
  Returns
    multi_key_dict: A multi-key dictionary of order documents,
                    keyed by Order_ID, Order_Tracking, and Order_Email.
  """
  documents = multi_key_dict()
  for order in json_documents:
    documents[order[ORDER_ID], order[ORDER_TRACKING], order[ORDER_EMAIL]] = order

  return documents


@tool("search_order_catalog",
      description="Search for orders with a key that is an order ID, tracking number, or email.")
def search_order_catalog(key: str, runtime: ToolRuntime):
  """
  Search for orders based on a key and return the matching order(s) from the multi-key dictionary.

  Args:
    key (str): The key to search for (Order_ID, Order_Tracking, or Order_Email).
    runtime (ToolRuntime): The runtime environment for the tool.
  """
  print(f"\n###searching orders by {key}###\n")
  print(f"\nstate: {runtime.state}")
  return runtime.context.orders.get(key, [])
