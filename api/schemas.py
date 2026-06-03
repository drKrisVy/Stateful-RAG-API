from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Strict schema for incoming user questions."""
    question: str = Field(..., min_length=3, description="The user's query.")

class ChatResponse(BaseModel):
    """Strict schema for outgoing AI answers."""
    answer: str
    is_cached: bool
    latency_ms: float = 0.0 