"""
Configuration Management
RUBRIC: Environment Setup & Configuration 
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration for Ribbon Chatbot"""

    # HuggingFace Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # FAISS Vector Store Configuration (legacy)
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

    # ChromaDB Vector Store Configuration
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_index")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ribbon_kb")

    # Azure OpenAI Configuration (kept for reference)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # Azure AI Search Configuration (kept for reference)
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "ribbon-kb-index")
    
    # Azure Storage (Optional)
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "ribbon-documents")
    
    # Azure Content Safety (Optional)
    AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")
    
    # Azure Monitor (Optional)
    APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ribbon-chatbot")
    
    # Ingestion Settings
    INGESTION_LIMIT = int(os.getenv("INGESTION_LIMIT", "0"))

    @classmethod
    def validate(cls):
        """Validate that required configuration values are present."""
        required = {
            "HF_TOKEN": cls.HF_TOKEN,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        return True
