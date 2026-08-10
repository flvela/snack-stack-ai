import os
import shutil
import time

from langchain_chroma import Chroma
from tools.config import get_embeddings  # Vector database for similarity search


def create_vector_store(persist_directory: str, documents: list, collection_name: str, overwrite: bool = False) -> Chroma:
    """
    Create a vector store from the provided documents.

    Args:
        persist_directory (str): Directory to persist the vector store.
        documents (list): List of documents to be added to the vector store.
        collection_name (str): Name of the collection in the vector store.
    Returns:
        vector_store: The created vector store.
    """
    if os.path.exists(persist_directory):
        if overwrite:
            try:
              shutil.rmtree(persist_directory)
            except PermissionError as e:
              # Windows may keep Chroma files locked briefly from a prior run.
              # Fall back to a unique directory so we can still create a vector store and not fail the run.
              persist_directory = f"{persist_directory}_{int(time.time())}"
              print(f"PermissionError encountered. Using a new persist directory: {persist_directory}")
        else:
            return Chroma(persist_directory=persist_directory, collection_name=collection_name)
          

    return Chroma.from_documents(documents=documents, 
                                 embedding=get_embeddings(), 
                                 persist_directory=persist_directory,
                                 collection_name=collection_name,
                                 collection_metadata={"hnsw:space": "cosine"}) # Use cosine distance for similarity search)


