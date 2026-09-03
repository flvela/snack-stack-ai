"""Unit tests for snack stack assistant"""
import pytest
from assistant import MENU_DIRECTORY, ORDERS_DIRECTORY, SnackStackAssistant
from tools.menu import load_menu_documents
from tools.orders import load_orders_documents


test_data = [
  # use menu agent test
  ("what are some italian dishes I can order",
   ["Italian dishes", "Margherita Pizza", "Pasta Primavera", "Aglio e Olio"]),
  # use order agent test
  ("What's the status of order ORD-201",
   ["ORD-201", "Out for Delivery"]),
  # use menu and order agent test
  ("Tell me about your Margherita Pizza and the status of my order with tracking number SS201TRK",
   ["ORD-201", "Out for Delivery", "Margherita Pizza"]),
  # interrupt assistant use case as no order key present
  ("what's the status of my order",
   ["ORD-201", "Out for Delivery"])
]


@pytest.mark.parametrize("query, expected_answer_strings", test_data)
def test_ask(query: str, expected_answer_strings: list[str]):
  """unit test for ask method of SnackStackAssistant"""
  assistant = SnackStackAssistant(orders=load_orders_documents(ORDERS_DIRECTORY),
                                  menu=load_menu_documents(MENU_DIRECTORY))
  answer = assistant.ask(query)
  print(answer)
  try:
    assert answer is not None
    if assistant.is_interrupted:
      answer = assistant.ask("ORD-201")
    for expected_string in expected_answer_strings:
      assert expected_string.lower() in answer.lower()
  finally:
    # shutdown cleaning up data store connection if test passes or fails
    assistant.shutdown()


def test_reset():
  """unit test for reset method of SnackStackAssistant"""
  assistant = SnackStackAssistant(orders=load_orders_documents(ORDERS_DIRECTORY),
                                  menu=load_menu_documents(MENU_DIRECTORY))
  thread_id = assistant.thread_id
  assistant.reset()
  try:
    assert thread_id != assistant.thread_id
  finally:
    # shutdown cleaning up data store connection if test passes or fails
    print("cleaning up assistant")
    assistant.shutdown()
