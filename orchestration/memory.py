from typing import TypedDict, List, Optional
from langchain_core.documents import Document

# 1. The strict data schema that moves through our pipeline
class EnterpriseState(TypedDict):
    question: str
    sub_queries: List[str]
    documents: List[Document]
    context: str
    answer: str
    is_cached: bool

# 2. Simulated Redis Cache
semantic_cache = {}

# Notice we changed 'str | None' to 'Optional[str]' here
def check_cache(query: str) -> Optional[str]:
    """Simulates a fast <5ms cache lookup."""
    return semantic_cache.get(query.lower().strip())

def update_cache(query: str, answer: str):
    """Simulates writing the generated answer to the cache."""
    semantic_cache[query.lower().strip()] = answer