import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models
LLM_MODEL = "llama-3.1-8b-instant" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

# Ingestion Limits
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Search Limits
RETRIEVER_K = 50  
FINAL_TOP_K = 3   
RRF_CONSTANT = 60