
from tools.menu import MENU_COLLECTION_NAME, load_menu_documents
from tools.vector_store import create_vector_store


def test_create_vector_store():
    persist_directory = 'data/chroma_store'
    documents = load_menu_documents('data/menu.json')
    vector_store = create_vector_store(persist_directory, documents, collection_name=MENU_COLLECTION_NAME, overwrite=True)
    test_chroma_store_data = [("Italian", 3), ("Vegan", 5)]
    for entry in test_chroma_store_data:
        query = entry[0]
        k = entry[1]
        results = vector_store.similarity_search(query, k=k)
        assert len(results) == k, f"Expected {k} results for query '{query}'"
    #cleanup
    vector_store.delete_collection()