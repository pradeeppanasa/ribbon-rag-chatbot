"""
Document Ingestion Pipeline
RUBRIC: Document Ingestion Pipeline 
- Ingestion script loads all documents 
- Documents are chunked properly 
- Batch indexing implemented 
- Index verification performed 

TASK: Ingest and index documents to Azure AI Search
"""
import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from src.search_engine import RibbonSearchEngine
from src.data_loader import RibbonDataLoader
from src.config import Config

import mlflow


def ingest_travel_documents():
    """
    Ingests Ribbon documents into FAISS vector store
    
    HINT: This function should:
    1. Initialize data loader and search engine
    2. Load all documents
    3. Split into chunks
    4. Batch index to Azure Search (batch_size=50)
    5. Verify with test query
    """
    print("\n🚀 Starting Ribbon Document Ingestion")
    print("=" * 70)

    # HINT: Initialize components
    loader = RibbonDataLoader()

    try:
        engine = RibbonSearchEngine()
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return

    # ====================
    # MLflow Setup (fail-safe)
    # ====================
    mlflow_active = False
    if Config.MLFLOW_TRACKING_URI:  
        try:
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)  
            mlflow.start_run(run_name="document_ingestion")  # HINT: "document_ingestion"
            mlflow_active = True
        except Exception as e:
            print(f"⚠️  MLflow disabled: {e}")

    try:
        # HINT: Load documents
        documents = loader.load_all_ribbon_documents()


        if not documents:
            print("\n⚠️  No documents found in data directory")
            print("\nExpected structure:")
            print("  data/")
            print("    ├── *.pdf   (policies, FAQs, rules)")
            print("    └── *.csv   (routes or tabular data)")
            return

        # HINT: Split into chunks
        chunks = loader.split_documents(documents)  

        print(f"\n📊 Ingestion Summary:")
        print(f"   Total chunks to index: {len(chunks)}")

        if mlflow_active:
            mlflow.log_param("total_chunks", len(chunks))
            mlflow.log_param("chunk_size", loader.text_splitter._chunk_size)
            mlflow.log_param("chunk_overlap", loader.text_splitter._chunk_overlap)

        # ====================
        # Wipe old index for a clean rebuild
        # ====================
        if os.path.exists(Config.FAISS_INDEX_PATH):
            shutil.rmtree(Config.FAISS_INDEX_PATH)
            print(f"🗑️  Removed stale index at '{Config.FAISS_INDEX_PATH}'")

        # ====================
        # Batch Ingestion
        # ====================
        print("\n📥 Indexing documents to FAISS local vector store...")
        batch_size = 50  # HINT: 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        ingested_count = 0
        failed_count = 0
        vector_store = None

        # HINT: Loop through chunks in batches
        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Indexing",
            total=total_batches
        ):
            batch = chunks[i:i + batch_size]

            try:
                if vector_store is None:
                    # Create fresh index from first batch
                    vector_store = FAISS.from_documents(batch, engine.embeddings)
                else:
                    # Append subsequent batches
                    vector_store.add_documents(batch)
                ingested_count += len(batch)
                time.sleep(0.5)  # avoid rate limits

            except Exception as e:
                print(f"\n❌ Error indexing batch {i // batch_size + 1}: {e}")
                failed_count += len(batch)

        # Persist and wire back to engine
        if vector_store:
            print("\n💾 Saving FAISS index to disk...")
            vector_store.save_local(Config.FAISS_INDEX_PATH)
            engine.vector_store = vector_store
            print(f"   FAISS index saved to '{Config.FAISS_INDEX_PATH}'")

        print(f"\n✅ Ingestion Complete!")
        print(f"   Successfully indexed: {ingested_count} chunks")
        if failed_count > 0:
            print(f"   Failed: {failed_count} chunks")

        if mlflow_active:
            mlflow.log_metric("ingested_count", ingested_count)
            mlflow.log_metric("failed_count", failed_count)

        # ====================
        # Verification
        # ====================
        print("\n🔍 Verifying index...")
        test_query = "What features does the Ribbon personal account offer?"
        results, _ = engine.search_by_text(test_query, k=3) 

        if results:
            print("✅ Index verification successful!")
            print(f"   Test query: '{test_query}'")
            print(f"   Retrieved: {len(results)} documents")
        else:
            print("⚠️  Warning: Test query returned no results")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")

    finally:
        if mlflow_active:
            mlflow.end_run() 

    print("\n" + "=" * 70)
    print("🎉 Ingestion pipeline completed!\n")


if __name__ == "__main__":
    ingest_travel_documents()