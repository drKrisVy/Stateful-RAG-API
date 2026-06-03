import json
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.documents import Document

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.hybrid_search import EnterpriseRetriever
from orchestration.memory import EnterpriseState, check_cache, update_cache

# Boot the LLM via Groq
llm = ChatGroq(
    model_name=config.LLM_MODEL, 
    api_key=config.GROQ_API_KEY, 
    temperature=0.1 # Low temp to prevent hallucinations
)

from retrieval.ingestion import ingest_enterprise_documents

# 1. Ingest real PDFs from the data/ folder
print("[System] Commencing Data Ingestion...")
corporate_documents = ingest_enterprise_documents("data")

# 2. Safe Fallback: If the user forgot to add a PDF, don't crash the server.
if not corporate_documents:
    print("[Warning] No PDFs found in data/. Booting with dummy state.")
    corporate_documents = [Document(page_content="System Initialized. Awaiting data in data/ folder.")]

# 3. Boot the Search Engine with REAL data
search_engine = EnterpriseRetriever(corporate_documents)

# --- THE GRAPH NODES ---

def cache_node(state: EnterpriseState):
    """Checks if we've answered this before."""
    cached_answer = check_cache(state["question"])
    if cached_answer:
        state["answer"] = cached_answer
        state["is_cached"] = True
    else:
        state["is_cached"] = False
    return state

def decompose_node(state: EnterpriseState):
    """Extracts keywords from complex questions."""
    # For now, we bypass the LLM call to save time/tokens and just use the raw question
    # We will expand this in the final production polish
    state["sub_queries"] = [state["question"]]
    return state

def retrieve_node(state: EnterpriseState):
    """Fires the FAISS+BM25 Search."""
    all_docs = []
    for query in state.get("sub_queries", [state["question"]]):
        docs = search_engine.retrieve(query)
        all_docs.extend(docs)
    
    # Deduplicate and extract text
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    state["documents"] = unique_docs
    state["context"] = "\n\n".join([d.page_content for d in unique_docs])
    return state

def generate_node(state: EnterpriseState):
    """Writes the final answer."""
    prompt = (
        "You are an enterprise AI assistant. Answer strictly based on the context.\n"
        "If the answer is not in the context, say 'Insufficient data.'\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt)
    state["answer"] = response.content
    
    update_cache(state["question"], state["answer"])
    return state

# --- THE ROUTING LOGIC ---

def route_after_cache(state: EnterpriseState):
    """Skip to the end if we got a cache hit."""
    if state["is_cached"]:
        return "end"
    return "decompose_node"

# --- COMPILE THE ENGINE ---
workflow = StateGraph(EnterpriseState)

workflow.add_node("cache_node", cache_node)
workflow.add_node("decompose_node", decompose_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("generate_node", generate_node)

workflow.set_entry_point("cache_node")
workflow.add_conditional_edges("cache_node", route_after_cache, {
    "end": END,
    "decompose_node": "decompose_node"
})
workflow.add_edge("decompose_node", "retrieve_node")
workflow.add_edge("retrieve_node", "generate_node")
workflow.add_edge("generate_node", END)

rag_engine = workflow.compile()
print("[System] Orchestrator Compiled.")