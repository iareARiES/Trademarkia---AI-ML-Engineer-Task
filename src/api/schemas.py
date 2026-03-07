"""
Pydantic request/response schemas for the API.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")


class DocumentResult(BaseModel):
    doc_id: str
    text: str
    label: str
    score: float


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    results: List[DocumentResult]
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class CacheDeleteResponse(BaseModel):
    message: str
    entries_removed: int
