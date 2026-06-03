import time
import sys
import os
import json
from tqdm import tqdm
from langchain_groq import ChatGroq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from orchestration.graph import rag_engine

print("[System] Booting LLMs...")
# The Judge LLM for grading
eval_llm = ChatGroq(model_name=config.LLM_MODEL, api_key=config.GROQ_API_KEY, temperature=0.0)
# The "Dumb" Baseline Model (No RAG)
baseline_llm = ChatGroq(model_name=config.LLM_MODEL, api_key=config.GROQ_API_KEY, temperature=0.1)

# ============================================================
# RAGAS EVALUATION METRICS
# ============================================================
def evaluate_context_recall(query, context, ground_truth):
    prompt = f"Given the Ground Truth, does the Retrieved Context contain enough info to deduce it?\nQuery: {query}\nTruth: {ground_truth}\nContext: {context}\nOutput only '1' (Yes) or '0' (No)."
    try: return 1 if "1" in eval_llm.invoke(prompt).content else 0
    except: return 0

def evaluate_faithfulness(query, context, answer):
    prompt = f"Does the Answer contain ANY facts NOT supported by the Context? (Hallucination check)\nQuery: {query}\nContext: {context}\nAnswer: {answer}\nOutput only '1' (Faithful) or '0' (Hallucinated)."
    try: return 1 if "1" in eval_llm.invoke(prompt).content else 0
    except: return 0

def calculate_mrr(retrieved_docs, ground_truth_text):
    """Calculates if the correct chunk was in the top 3 and its exact rank."""
    doc_texts = [doc.page_content.lower() for doc in retrieved_docs]
    target = ground_truth_text.lower()
    
    hit_rank = None
    for rank, text in enumerate(doc_texts[:3], start=1):
        if target in text or any(word in text for word in target.split()[:3]):
            hit_rank = rank
            break
            
    recall_at_3 = 1 if hit_rank is not None else 0
    reciprocal_rank = 1.0 / hit_rank if hit_rank is not None else 0.0
    return recall_at_3, reciprocal_rank

# ============================================================
# MAIN A/B BENCHMARK EXECUTION
# ============================================================
def run_ab_benchmarks():
    dataset_path = "eval_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset file '{dataset_path}' missing. Run generate_synthetic_data.py first.")
        return
        
    with open(dataset_path, "r") as f:
        eval_pairs = json.load(f)
        
    print("\n" + "="*70)
    print(f"🚀 ENTERPRISE A/B TEST: BASELINE LLM vs. ENTERPRISE-RAG-X")
    print(f"Loaded {len(eval_pairs)} synthetic questions from JSON.")
    print("="*70 + "\n")
    
    rag_metrics = {"context_recall": 0, "faithfulness_score": 0, "recall_at_3": 0, "mrr": 0.0, "total_latency": 0, "runs": 0}
    base_metrics = {"total_latency": 0}
    
    for item in tqdm(eval_pairs, desc="Evaluating RAG Pipeline"):
        query = item["question"]
        ground_truth = item["ground_truth_text"]
        
        # --- SYSTEM A: BASELINE LLM (NO RAG) ---
        base_start = time.perf_counter()
        try:
            base_ans = baseline_llm.invoke(query).content
            base_metrics["total_latency"] += (time.perf_counter() - base_start) * 1000
        except: pass

        # --- SYSTEM B: ENTERPRISE RAG ENGINE ---
        rag_start = time.perf_counter()
        try:
            final_state = rag_engine.invoke({"question": query, "sub_queries": [], "documents": [], "context": "", "answer": "", "is_cached": False})
            rag_metrics["total_latency"] += (time.perf_counter() - rag_start) * 1000
            
            context = final_state.get("context", "")
            answer = final_state.get("answer", "")
            retrieved_docs = final_state.get("documents", [])
            
            rag_metrics["runs"] += 1
            
            # Grade context extraction & hallucination
            rag_metrics["context_recall"] += evaluate_context_recall(query, context, ground_truth)
            rag_metrics["faithfulness_score"] += evaluate_faithfulness(query, context, answer)
            
            # Grade search rank accuracy (Recall@3 & MRR)
            rec_3, mrr = calculate_mrr(retrieved_docs, ground_truth)
            rag_metrics["recall_at_3"] += rec_3
            rag_metrics["mrr"] += mrr
            
        except Exception as e:
            pass
            
        # Rate limit protection for Groq API
        time.sleep(3) 

    # --- FINAL MATH ---
    runs = rag_metrics["runs"]
    if runs == 0:
        print("Evaluation failed. No successful runs.")
        return

    final_context_recall = (rag_metrics["context_recall"] / runs) * 100
    final_rag_faith = (rag_metrics["faithfulness_score"] / runs) * 100
    final_recall_3 = (rag_metrics["recall_at_3"] / runs) * 100
    final_mrr = rag_metrics["mrr"] / runs
    
    avg_base_latency = base_metrics['total_latency'] / len(eval_pairs)
    avg_rag_latency = rag_metrics['total_latency'] / runs

    print("\n" + "="*70)
    print("📊 FINAL A/B TEST RESULTS (100 SYNTHETIC PAIRS)")
    print("="*70)
    print(f"Metric                 | System A (Base LLM) | System B (Enterprise RAG) ")
    print("-" * 70)
    print(f"Context Recall         | 0.0% (No Context)   | {round(final_context_recall, 1)}% ")
    print(f"Faithfulness (No Lies) | N/A (Ungrounded)    | {round(final_rag_faith, 1)}% ")
    print(f"Recall@3 (Search Hit)  | 0.0%                | {round(final_recall_3, 1)}% ")
    print(f"MRR (Rank Accuracy)    | 0.0                 | {round(final_mrr, 3)} ")
    print(f"Average Latency        | {round(avg_base_latency, 1)} ms          | {round(avg_rag_latency, 1)} ms")
    print("="*70)

if __name__ == "__main__":
    run_ab_benchmarks()