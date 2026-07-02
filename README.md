# LLMOps Stateful RAG API

A production-grade, stateful Retrieval-Augmented Generation (RAG) system engineered to answer complex financial queries over SEC documents. The pipeline integrates a hybrid search engine for maximum context recall, an automated LLM-as-a-judge evaluation harness to mathematically verify output correctness, and a Prometheus/Grafana infrastructure for real-time traffic and latency telemetry.

---

## Results

| Metric | Score |
| :--- | :--- |
| **Hybrid Search Recall@3** | 98.9% |
| **Mean Reciprocal Rank (MRR)** | 0.965 |
| **Context Retrieval Recall (Evaluation Dataset)** | 86.3% |
| **Repeated-Query Cache Latency** | < 5ms |
| **Base Pipeline End-to-End Latency** | ~5s average |
| **Evaluation Suite Scope** | 100 synthetic questions |

---

## Overview

*   **Stateful Orchestration:** Built using FastAPI and LangGraph to manage conversational state and enforce strict, deterministic execution graphs between the user, retrieval steps, and the LLM.
*   **Hybrid Retrieval Engine:** Combines dense semantic vector search via FAISS (`text-embedding-3-small`) with sparse keyword retrieval via BM25. Results are fused using Reciprocal Rank Fusion (RRF) to hit **98.9% Recall@3**, ensuring highly precise financial context extraction.
*   **Redis Semantic Caching:** Intercepts repeated or semantically identical incoming user queries via a Redis semantic cache layer. Hits bypass the vector database and LLM entirely, crashing repeated-query latency from ~5s down to **< 5ms** while cutting token costs.
*   **Automated LLM-as-a-Judge Evaluation:** Features an isolated execution script (`generate_synthetic_data.py`) to generate a 100-question test suite containing ground-truth targets from SEC documents. Pipeline iterations are strictly benchmarked on faithfulness, relevancy, and context recall using an LLM-as-a-judge evaluation harness.
*   **Production Telemetry Stack:** Instrumented with `prometheus-fastapi-instrumentator` and custom Prometheus counters to track API traffic volumes, response code distributions, latency histograms, and real-time Redis cache hit/miss rates via localized Grafana dashboards.

---

## Tech Stack

**Python** · **FastAPI** · **LangGraph** · **FAISS** · **BM25** · **Redis** · **Prometheus** · **Grafana** · **Docker**

---

## Project Structure

```text
enterprise-rag-x/
├── api/                        # FastAPI routers, endpoints, and application setup
├── orchestration/              # LangGraph multi-step state and execution workflows
├── retrieval/                  # FAISS dense search, BM25 sparse search, and Redis caching logic
├── config.py                   # Central environment variable management and security configs
├── Dockerfile                  # Container blueprint for python environment execution
├── docker-compose.yml          # Multi-container orchestration (FastAPI, Redis, Prometheus, Grafana)
├── eval_dataset.json           # Curated 100-question benchmark dataset with ground-truth pairs
├── generate_synthetic_data.py  # Script generating synthetic evaluation distributions
├── prometheus.yml              # Scrape target rules and scrape interval configurations for telemetry
├── requirements.txt            # System dependencies
├── run_benchmarks.py           # Automated evaluation harness running LLM-as-a-judge
├── .gitignore                  # Git tracking exclusions for environment files and datasets
└── README.md                   # Repository documentation
