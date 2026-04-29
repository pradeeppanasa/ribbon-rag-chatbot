"""
Vector Store Configuration
RUBRIC: Vector Store & RAG Setup
- ChromaDB vector store initialized correctly
- Azure OpenAI embeddings configured properly
- Direct chromadb client (avoids heavy langchain_community import at startup)

TASK: Initialize ChromaDB local vector store with Azure OpenAI embeddings
"""
import os
from src.config import Config


class _Doc:
    """Minimal document matching langchain Document interface (page_content, metadata)."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata or {}


class ChromaVectorStore:
    """
    Direct chromadb wrapper that matches the langchain vectorstore interface.
    Uses chromadb Python client directly to avoid the heavy langchain_community import.
    """

    def __init__(self, embedding_function, persist_dir, collection_name):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_collection(collection_name)
        self._embed = embedding_function  # LangChain AzureOpenAIEmbeddings instance

    def similarity_search(self, query: str, k: int = 5):
        count = self._collection.count()
        if count == 0:
            return []
        embedding = self._embed.embed_query(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, count),
            include=["documents", "metadatas"],
        )
        return [
            _Doc(text, meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def add_documents(self, documents):
        """Used by ingestion to add documents (delegates to langchain_community for ingestion)."""
        raise NotImplementedError("Use ingestion.py for adding documents.")


def get_vector_store(embedding_function):
    """
    Returns ChromaDB vector store loaded from disk, or None if not yet built.

    Args:
        embedding_function: LangChain AzureOpenAIEmbeddings instance
    """
    persist_dir = Config.CHROMA_PATH
    collection_name = Config.CHROMA_COLLECTION

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        store = ChromaVectorStore(embedding_function, persist_dir, collection_name)
        print(f"Loaded ChromaDB index from '{persist_dir}'")
        return store

    print(f"No ChromaDB index found at '{persist_dir}' — run ingestion first.")
    return None
