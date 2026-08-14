"""unit tests for VectorStore"""
import shutil

from chromadb.api import ClientAPI
from tools.menu import MENU_COLLECTION_NAME, load_menu_documents
from tools.vector_store import PERSIST_DIRECTORY, VectorStore

def test_constructor():
  """contructor test"""
  vector_store = VectorStore(PERSIST_DIRECTORY, MENU_COLLECTION_NAME)
  assert vector_store.persist_directory == PERSIST_DIRECTORY
  assert vector_store.collection_name == MENU_COLLECTION_NAME
  assert isinstance(vector_store.client, ClientAPI)

def test_create_vector_store():
  """test_create_vector_store test"""
  vector_store = VectorStore(PERSIST_DIRECTORY, MENU_COLLECTION_NAME)
  documents = load_menu_documents('data/menu.json')
  collection = vector_store.get_create_collection(documents)
  test_chroma_store_data = [("Italian", 3), ("Vegan", 5)]
  for entry in test_chroma_store_data:
    query = entry[0]
    k = entry[1]
    results = collection.similarity_search(query, k=k)
    assert len(results) == k, f"Expected {k} results for query '{query}'"
  #cleanup
  vector_store.delete_collection()
  vector_store.close()
  shutil.rmtree(PERSIST_DIRECTORY)
