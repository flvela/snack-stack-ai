"""tools to load menu documents and search menu catalogue"""
import json

from langchain_core.documents import Document
from langchain.tools import ToolRuntime, tool


# MENU CONSTANTS
MENU_DISH = "Dish"
MENU_CUISINE = "Cuisine"
MENU_PRICE = "Price"
MENU_RATING = "Rating"
MENU_DIETARY = "Dietary"
MENU_DESCRIPTION = "Description"
MENU_PERSIST_DIRECTORY = "data/chroma_store"
MENU_COLLECTION_NAME = "menu_collection"


def load_menu_documents(file_path: str):
  """
  Load menu documents from a JSON file.

  Args:
    file_path (str): Path to the JSON file containing menu documents.
  Returns:
    list: A list of menu documents.
  """
  with open(file_path, 'r', encoding='utf-8') as file:
    items = json.load(file)

    documents = []
    for item in items:
      content = f"""
        Dish: {item[MENU_DISH]}
        Cuisine: {item[MENU_CUISINE]}
        Price: {item[MENU_PRICE]}
        Rating: {item[MENU_RATING]}
        Dietary: {item[MENU_DIETARY]}
        Description: {item[MENU_DESCRIPTION]}""".strip()

      doc = Document(page_content=content,
                     metadata={
                       MENU_DISH: item[MENU_DISH],
                       MENU_CUISINE: item[MENU_CUISINE],
                       MENU_DIETARY: item[MENU_DIETARY],
                       })
      documents.append(doc)

    return documents

@tool("search_menu_catalog", description="Search for menu items based on a query.")
def search_menu_catalog(query: str, runtime: ToolRuntime):
  """
  Search for menu items based on a query and return the top 3 most similar items
  from the vector store.

  Args:
    query (str): The search query.
    runtime (ToolRuntime): The runtime environment for the tool.
  """
  print(f"\n### searching menu by {query}###\n")
  print(f"\nstate: {runtime.state}")
  documents = runtime.context.menu_collection.similarity_search(query, k=3)
  if not documents:
    return []

  output = f'Top {len(documents)} matches for {query}:\n\n'
  for document in documents:
    output += document.page_content + "\n---\n"
  return output
