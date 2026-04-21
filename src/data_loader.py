"""
Data Loader for Ribbon Documents
RUBRIC: Data Loading & Preparation 
- RibbonDataLoader implemented correctly
- PDFs loaded directly from data/ folder 
- Document categorization based on filename
- Text chunking configured properly 

TASK: Load PDFs from data/ folder, categorize them, and chunk them
"""
import os
import pandas as pd
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# HINT: Data directory should point to "data"
DATA_DIR = Path("data")  # HINT: Path("data")


class RibbonDataLoader:
    """Loads Ribbon documents (PDFs, CSVs) from data directory"""

    def __init__(self):
        # HINT: Initialize text splitter with chunk_size=1000, chunk_overlap=200
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # HINT: 1000
            chunk_overlap=200,  # HINT: 200
            separators=["\n\n", "\n", " ", ""]    
        )

    def load_pdfs_from_data_directory(self) -> List[Document]:
        """
        Load all PDFs directly from data/ directory
        
        HINT: This method should:
        1. Check if DATA_DIR exists
        2. Get all PDF files using glob
        3. Loop through files and load with PyPDFLoader
        4. Add metadata (source, file_type)
        5. Return list of documents
        """
        documents = []

        # HINT: Check if DATA_DIR exists
        if not DATA_DIR.exists():  
            print(f"Warning: Directory {DATA_DIR} does not exist")
            return documents

        # HINT: Get all PDF files using glob pattern "*.pdf"
        pdf_files = list(DATA_DIR.glob("*.pdf")) 
        print(f"Found {len(pdf_files)} PDF files in data/")

        for pdf_file in pdf_files:
            try:
                # HINT: Use PyPDFLoader to load the PDF
                loader = PyPDFLoader(str(pdf_file))  # HINT: PyPDFLoader
                docs = loader.load() 

                # HINT: Add metadata to each document
                for doc in docs:
                    doc.metadata.update({
                        'source': pdf_file.name,
                        'file_type': "pdf",  
                       'category': self._categorize_document(pdf_file.name)
                    })

                documents.extend(docs)
                print(f"  ✓ Loaded: {pdf_file.name} ({len(docs)} pages)")

            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")

        return documents
    
    def _categorize_document(self, filename: str) -> str:
        """Categorize document based on filename"""
        filename_lower = filename.lower()
        
        if "air-india" in filename_lower or "ai-schedule" in filename_lower:
            return "air_india_policies"
        elif "u.s. department" in filename_lower or "transportation" in filename_lower:
            return "us_dot_regulations"
        elif "booking" in filename_lower or "policy" in filename_lower:
            return "booking_policies"
        elif "refund" in filename_lower:
            return "refund_policies"
        elif "privacy" in filename_lower:
            return "privacy_policies"
        else:
            return "general"

    def load_csvs_from_data_directory(self) -> List[Document]:
        """
        Load all CSVs directly from data/ directory
        
        HINT: Similar to PDF loading but for CSV files
        """
        documents = []

        if not DATA_DIR.exists():
            print(f"Warning: Directory {DATA_DIR} does not exist")
            return documents

        # HINT: Get all CSV files
        csv_files = list(DATA_DIR.glob("*.csv")) 
        print(f"Found {len(csv_files)} CSV files in data/")

        for csv_file in csv_files:
            try:
                # HINT: Use pandas to read CSV
                df = pd.read_csv(str(csv_file))

                # HINT: Convert each row to a Document
                for idx, row in df.iterrows():  # HINT: iterrows()
                    # HINT: Create content by joining column:value pairs
                    content = " | ".join(
                        f"{col}: {val}" for col, val in row.items() 
                    )

                    # HINT: Create Document with metadata
                    documents.append(
                        Document(
                            page_content=content,  
                            metadata={
                                'source': csv_file.name, 
                                'file_type': "csv",  
                                'row_index': idx 
                            }
                        )
                    )

                print(f"  ✓ Loaded: {csv_file.name} ({len(df)} rows)")

            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")

        return documents

    def load_all_ribbon_documents(self) -> List[Document]:
        """
        Load all PDFs and CSVs from Ribbon data directory
        
        HINT: Call both PDF and CSV loaders and combine results
        """
        all_documents = []

        print("\n📂 Loading Ribbon knowledge base...")
        print("=" * 60)

        # HINT: Load PDFs
        pdf_docs = self.load_pdfs_from_data_directory() 
        all_documents.extend(pdf_docs)

        # HINT: Load CSVs
        csv_docs = self.load_csvs_from_data_directory()
        all_documents.extend(csv_docs)

        print("=" * 60)
        print(f"✅ Total documents loaded: {len(all_documents)}")

        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        HINT: Use self.text_splitter.split_documents()
        """
        print(f"\n✂️  Splitting {len(documents)} documents into chunks...")

        # HINT: Use self.text_splitter to split documents
        chunks = self.text_splitter.split_documents(documents)

        print(f"✅ Created {len(chunks)} chunks")
        print(
            f"   Average chunk size: "
            f"{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars"
        )

        return chunks


if __name__ == "__main__":
    loader = RibbonDataLoader()
    docs = loader.load_all_ribbon_documents()
    chunks = loader.split_documents(docs)

    print("\n📊 Summary:")
    print(f"   Total chunks: {len(chunks)}")