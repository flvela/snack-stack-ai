from tools.menu import MENU_CUISINE, MENU_DIETARY, MENU_DISH, load_menu_documents


def test_load_menu_documents():
  """unit tests for load_menu_documents"""
  file_path = 'data/menu.json'
  documents = load_menu_documents(file_path)
  assert len(documents) == 8, "Expected 8 menu documents"
  for doc in documents:
    assert doc.metadata.get(MENU_DISH) is not None, "Dish metadata should not be None"
    assert doc.metadata.get(MENU_CUISINE) is not None, "Cuisine metadata should not be None"
    assert doc.metadata.get(MENU_DIETARY) is not None, "Dietary metadata should not be None"
