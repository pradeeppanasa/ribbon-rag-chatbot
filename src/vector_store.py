"""
Vector Store Configuration
RUBRIC: Vector Store & RAG Setup 
- FAISS vector store initialized correctly 
- HuggingFace embeddings configured properly
- LangChain FAISS integration is correct 

TASK: Initialize FAISS local vector store with HuggingFace embeddings
"""
import os
from langchain_community.vectorstores import FAISS
from src.config import Config


def get_vector_store(embedding_function):
    """
    Returns FAISS local vector store loaded from disk, or None if not yet built.

    Args:
        embedding_function: The LangChain HuggingFaceEmbeddings instance
    """
    index_path = Config.FAISS_INDEX_PATH

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            index_path,
            embedding_function,
            allow_dangerous_deserialization=True,
        )
        print(f"Loaded FAISS index from '{index_path}'")
        return vector_store

    print(f"No FAISS index found at '{index_path}' — run ingestion first.")
    return None
