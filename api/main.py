import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import ChatRequest, ChatResponse
from orchestration.graph import rag_engine

# Boot the FastAPI Server
app = FastAPI(
    title="EnterpriseRAG-X API",
    description="Stateful Retrieval Engine for Corporate Data",
    version="1.0.0"
)

# Allow external apps to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PROMETHEUS TELEMETRY METRICS (LLMWatch Integration)
# ============================================================
REQUEST_COUNT = Counter(
    "rag_requests_total", 
    "Total RAG API requests processed", 
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", 
    "Latency of RAG API executions in seconds", 
    ["endpoint"]
)

CACHE_HITS = Counter(
    "rag_cache_hits_total", 
    "Total times the semantic Redis cache was hit"
)

# Expose the raw metrics stream for Prometheus to scrape
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/health")
async def health_check():
    """Simple endpoint to prove the server is running."""
    return {"status": "online", "engine": "LangGraph + FAISS + BM25"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """The main endpoint that passes questions to the AI with real-time metrics tracking."""
    start_time = time.perf_counter()
    
    try:
        # 1. Define the starting state for LangGraph
        initial_state = {
            "question": request.question,
            "sub_queries": [],
            "documents": [],
            "context": "",
            "answer": "",
            "is_cached": False
        }
        
        # 2. Execute the State Machine
        final_state = rag_engine.invoke(initial_state)
        
        # 3. Calculate latency
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        latency_seconds = (time.perf_counter() - start_time)
        
        # 4. Telemetry Tracking
        if final_state.get("is_cached", False):
            CACHE_HITS.inc()
            
        REQUEST_COUNT.labels(endpoint="/chat", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/chat").observe(latency_seconds)
        
        # 5. Return the strict JSON response
        return ChatResponse(
            answer=final_state["answer"],
            is_cached=final_state["is_cached"],
            latency_ms=latency_ms
        )
        
    except Exception as e:
        print(f"[Error] Pipeline Failure: {str(e)}")
        REQUEST_COUNT.labels(endpoint="/chat", status="error").inc()
        raise HTTPException(status_code=500, detail="Internal AI Engine Error")