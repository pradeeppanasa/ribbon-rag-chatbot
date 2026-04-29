"""
Ribbon Search Engine with RAG
RUBRIC: Search Engine Implementation
- RibbonSearchEngine initialized correctly
- search_by_text performs similarity search
- synthesize_response generates grounded answers
- Governance checks integrated

TASK: Implement RAG search engine with governance integration
"""
from src.config import Config
from governance.governance_gate import GovernanceGate
from src.vector_store import get_vector_store


class RibbonSearchEngine:
    """RAG-powered search engine for Ribbon queries"""

    def __init__(self):
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # lazy: deferred for fast startup

        # Initialize governance gate
        self.governance_gate = GovernanceGate()

        # Initialize Azure Chat OpenAI LLM
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0,
            max_tokens=300,
        )

        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )

        # Initialize ChromaDB vector store
        self.vector_store = get_vector_store(self.embeddings)

    def search_by_text(self, query_text: str, k: int = 5):
        mlflow_active = False
        if Config.MLFLOW_TRACKING_URI:
            try:
                import mlflow  # lazy: only when tracking URI is configured
                mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
                mlflow_active = True
            except Exception:
                pass

        def _search():
            print(f"DEBUG: Text Query: {query_text}")

            gov_check = self.governance_gate.validate_input(query_text)
            if not gov_check['passed']:
                return [], "Query blocked by security checks."

            if mlflow_active:
                try:
                    mlflow.log_param("k", k)
                    mlflow.log_param("query_text", query_text)
                except Exception:
                    pass

            if self.vector_store is None:
                return [], "Knowledge base not ready — please run ingestion first."

            docs = self.vector_store.similarity_search(query_text, k=k)

            if mlflow_active:
                try:
                    mlflow.log_metric("results_count", len(docs))
                except Exception:
                    pass

            return docs, query_text

        if mlflow_active:
            with mlflow.start_run(run_name="search_ribbon_info", nested=True):
                return _search()
        else:
            return _search()

    def synthesize_response(self, docs, user_query):
        mlflow_active = False
        if Config.MLFLOW_TRACKING_URI:
            try:
                import mlflow  # lazy: only when tracking URI is configured
                mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
                mlflow_active = True
            except Exception:
                pass

        def _synthesize():
            if not docs:
                return "I am sorry, I could not find relevant information. Please try rephrasing or contact support."

            context = "\n".join([
                f"- {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})"
                for doc in docs
            ])

            prompt = f"""
            You are a helpful AI assistant for Ribbon, a financial services platform.
            Use the following information from our knowledge base to answer the customer's question.

            Knowledge Base Information:
            {context}

            Customer Question: "{user_query}"

            Please provide a clear, helpful, and accurate answer based on the information above.
            If the information is not sufficient, let the customer know and provide general guidance.
            """

            response = self.llm.invoke(prompt).content

            gov_check = self.governance_gate.validate_output(response)
            if not gov_check['passed']:
                return "I generated a response but it didn't pass safety checks. Please rephrase your question."

            if mlflow_active:
                try:
                    mlflow.log_text(response, "final_response.txt")
                except Exception:
                    pass

            return response

        if mlflow_active:
            with mlflow.start_run(run_name="synthesize_response", nested=True):
                return _synthesize()
        else:
            return _synthesize()
