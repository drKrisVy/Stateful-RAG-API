from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# Import our centralized settings from the root folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class EnterpriseRetriever:
    def __init__(self, documents):
        """Initializes the heavy search models."""
        print("[System] Initializing Dense Embeddings (FAISS)...")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        
        print("[System] Initializing Sparse Keywords (BM25)...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = config.RETRIEVER_K
        
        print(f"[System] Booting Local Neural Reranker: {config.RERANKER_MODEL}...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def reciprocal_rank_fusion(self, dense_results, sparse_results):
        """Mathematically blends vector and keyword ranks using RRF."""
        rrf_scores = {}
        k = config.RRF_CONSTANT
        
        # Grade the Dense (Vector) Results
        for rank, doc in enumerate(dense_results):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            rrf_scores[doc.page_content]["score"] += 1.0 / (rank + k)
            
        # Grade the Sparse (Keyword) Results
        for rank, doc in enumerate(sparse_results):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            rrf_scores[doc.page_content]["score"] += 1.0 / (rank + k)
            
        # Sort by the highest combined score and slice to our candidate pool limit
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items][:config.RETRIEVER_K]

    def retrieve(self, query: str):
        """Executes the full pipeline: FAISS + BM25 -> RRF -> Cross-Encoder."""
        # 1. Parallel Retrieval
        dense_docs = self.faiss_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)
        
        # 2. Mathematical Fusion
        fused_candidates = self.reciprocal_rank_fusion(dense_docs, sparse_docs)
        if not fused_candidates:
            return []
            
        # 3. Neural Reranking (The Hallucination Killer)
        pairs = [[query, doc.page_content] for doc in fused_candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort documents by their neural network score and return the absolute top K
        ranked_docs = sorted(zip(scores, fused_candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked_docs[:config.FINAL_TOP_K]]