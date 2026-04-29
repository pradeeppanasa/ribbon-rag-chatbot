"""
Document Ingestion Pipeline
RUBRIC: Document Ingestion Pipeline
- Ingestion script loads all documents
- Documents are chunked properly
- Batch indexing implemented
- Index verification performed

TASK: Ingest and index documents to ChromaDB
"""
import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from src.search_engine import RibbonSearchEngine
from src.data_loader import RibbonDataLoader
from src.config import Config

import mlflow


def ingest_travel_documents():
    """
    Ingests Ribbon documents into ChromaDB vector store
    """
    print("\n🚀 Starting Ribbon Document Ingestion")
    print("=" * 70)

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
            mlflow.start_run(run_name="document_ingestion")
            mlflow_active = True
        except Exception as e:
            print(f"⚠️  MLflow disabled: {e}")

    try:
        documents = loader.load_all_ribbon_documents()

        if not documents:
            print("\n⚠️  No documents found in data directory")
            return

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
        if os.path.exists(Config.CHROMA_PATH):
            shutil.rmtree(Config.CHROMA_PATH)
            print(f"🗑️  Removed stale index at '{Config.CHROMA_PATH}'")

        # ====================
        # Batch Ingestion
        # ====================
        print("\n📥 Indexing documents to ChromaDB...")
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        ingested_count = 0
        failed_count = 0
        vector_store = None

        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing", total=total_batches):
            batch = chunks[i:i + batch_size]
            try:
                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        batch,
                        engine.embeddings,
                        collection_name=Config.CHROMA_COLLECTION,
                        persist_directory=Config.CHROMA_PATH,
                    )
                else:
                    vector_store.add_documents(batch)
                ingested_count += len(batch)
                time.sleep(0.5)
            except Exception as e:
                print(f"\n❌ Error indexing batch {i // batch_size + 1}: {e}")
                failed_count += len(batch)

        if vector_store:
            engine.vector_store = vector_store  # Chroma auto-persists
            print(f"   ChromaDB index saved to '{Config.CHROMA_PATH}'")

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
