"""
Ribbon Search Engine with RAG
RUBRIC: Search Engine Implementation 
- RibbonSearchEngine initialized correctly 
- search_by_text performs similarity search 
- synthesize_response generates grounded answers 
- Governance checks integrated 

TASK: Implement RAG search engine with governance integration
"""
import mlflow
from langchain_huggingface import HuggingFaceEmbeddings
from src.hf_llm import HFInferenceChat
from src.config import Config
from governance.governance_gate import GovernanceGate
from src.vector_store import get_vector_store

class RibbonSearchEngine:
    """RAG-powered search engine for Ribbon queries"""
    
    def __init__(self):
        """
        Initialize search engine components
        
        HINT: Initialize:
        1. Governance gate
        2. Azure Chat OpenAI (LLM)
        3. Azure OpenAI Embeddings
        4. Vector store using get_vector_store
        """
        # HINT: Initialize governance gate
        self.governance_gate = GovernanceGate()

        # Initialize HuggingFace LLM via InferenceClient (provider='auto')
        self.llm = HFInferenceChat(
            model_id=Config.HF_LLM_MODEL,
            hf_token=Config.HF_TOKEN,
            max_tokens=512,
            temperature=0.1,
        )

        # Initialize HuggingFace Embeddings (runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.HF_EMBEDDING_MODEL,
        )

        # HINT: Initialize Vector Store using get_vector_store function
        self.vector_store = get_vector_store(self.embeddings)
    
    def search_by_text(self, query_text: str, k: int = 5):
        """
        Search for Ribbon information using a text query
        
        HINT: This method should:
        1. Set MLflow experiment
        2. Start MLflow run
        3. Validate input with governance gate
        4. Perform similarity search on vector store
        5. Log metrics to MLflow
        6. Return results and query
        """
        
        # HINT: Set MLflow experiment name from Config
        mlflow_active = False
        if Config.MLFLOW_TRACKING_URI:
            try:
                mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
                mlflow_active = True
            except Exception:
                pass

        def _search():
            print(f"DEBUG: Text Query: {query_text}")

            # HINT: Validate input using governance gate
            gov_check = self.governance_gate.validate_input(query_text)

            if not gov_check['passed']:
                return [], "Query blocked by security checks."

            if mlflow_active:
                try:
                    mlflow.log_param("k", k)
                    mlflow.log_param("query_text", query_text)
                except Exception:
                    pass

            # HINT: Perform similarity search on vector store
            if self.vector_store is None:
                return [], "Knowledge base not ready — please run ingestion first."
            # MMR gives diverse results — avoids returning near-duplicate chunks
            raw_docs = self.vector_store.max_marginal_relevance_search(
                query_text, k=k * 3, fetch_k=50
            )
            # Deduplicate by source file — keep best chunk per PDF
            seen_sources = set()
            docs = []
            for doc in raw_docs:
                src = doc.metadata.get("source", "")
                if src not in seen_sources:
                    seen_sources.add(src)
                    docs.append(doc)
                if len(docs) >= k:
                    break

            if mlflow_active:
                mlflow.log_metric("results_count", len(docs))

            return docs, query_text

        if mlflow_active:
            with mlflow.start_run(run_name="search_travel_info", nested=True):
                return _search()
        else:
            return _search()

    def synthesize_response(self, docs, user_query):
        """
        Generate a conversational response based on retrieved documents
        
        HINT: This method should:
        1. Start MLflow run
        2. Build context from retrieved documents
        3. Create a prompt for the LLM
        4. Generate response using LLM
        5. Validate output with governance gate
        6. Log response to MLflow
        7. Return final response
        """
        
        mlflow_active = False
        if Config.MLFLOW_TRACKING_URI:
            try:
                mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
                mlflow_active = True
            except Exception:
                pass

        def _synthesize():
            # HINT: Handle case when no documents found
            if not docs:
                return "I am sorry, I could not find relevant information. Please try rephrasing or contact support."

            # HINT: Build context from documents
            context = "\n".join([
                f"- {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})"
                for doc in docs
            ])

            # HINT: Create prompt for LLM
            prompt = f"""
            You are a helpful AI assistant for Ribbon, a financial services platform.
            Use the following information from our knowledge base to answer the customer's question.

            Knowledge Base Information:
            {context}

            Customer Question: "{user_query}"

            Please provide a clear, helpful, and accurate answer based on the information above.
            If the information is not sufficient, let the customer know and provide general guidance.
            """

            # HINT: Generate response using LLM
            response = self.llm.invoke(prompt).content

            # HINT: Validate output using governance gate
            gov_check = self.governance_gate.validate_output(response)

            if not gov_check['passed']:
                return "I generated a response but it didn't pass safety checks. Please rephrase your question."

            if mlflow_active:
                try:
                    mlflow.log_text(response, "final_response.txt")
                except Exception:
                    pass  # ignore encoding errors on Windows

            return response

        if mlflow_active:
            with mlflow.start_run(run_name="synthesize_response", nested=True):
                return _synthesize()
        else:
            return _synthesize()