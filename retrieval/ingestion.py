import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import centralized settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def ingest_enterprise_documents(data_directory: str = "data"):
    """
    Scans the data directory for PDFs, extracts text, and applies 
    mathematical chunking with semantic overlap.
    """
    all_chunks = []
    
    # 1. Configure the text splitter using our config thresholds
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,       # 800 tokens
        chunk_overlap=config.CHUNK_OVERLAP, # 150 token overlap to prevent cut-off sentences
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try to split at paragraphs first
    )
    
    # Ensure the directory exists
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"[Warning] Created '{data_directory}' directory. Please add PDFs.")
        return []

    # 2. Iterate through all PDFs in the data folder
    for filename in os.listdir(data_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_directory, filename)
            print(f"[Ingestion] Processing: {filename}...")
            
            try:
                loader = PyPDFLoader(file_path)
                raw_documents = loader.load()
                
                chunked_documents = text_splitter.split_documents(raw_documents)
                all_chunks.extend(chunked_documents)
                print(f"   -> Extracted {len(chunked_documents)} semantic chunks.")
            except Exception as e:
                print(f"[Error] Failed to process {filename}: {str(e)}")
                
    return all_chunks