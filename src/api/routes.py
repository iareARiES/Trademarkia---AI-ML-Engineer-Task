"""
API routes: POST /query, GET /cache/stats, DELETE /cache.
"""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CacheDeleteResponse,
    CacheStatsResponse,
    DocumentResult,
    QueryRequest,
    QueryResponse,
)
from src.api.state import compute_memberships, state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Semantic search with cache-aware retrieval.

    1. Encode query → embedding
    2. Compute FCM memberships
    3. Cache lookup
    4. On miss: query ChromaDB, insert to cache
    """
    if state.model is None or state.collection is None or state.cache is None:
        raise HTTPException(status_code=503, detail="Service not ready. Models not loaded.")

    query_text = request.query.strip()

    # 1. Encode query
    query_embedding = state.model.encode(
        [query_text],
        normalize_embeddings=True,
    )[0].astype(np.float32)

    # 2. Compute FCM memberships
    query_pca = state.pca_model.transform(query_embedding.reshape(1, -1))[0]
    memberships = compute_memberships(
        query_pca,
        state.cluster_centers,
        m=state.fcm_meta.get('m', 2.0),
    )
    dominant_cluster = int(np.argmax(memberships))

    # 3. Cache lookup
    cache_result = state.cache.lookup(query_embedding, memberships)

    if cache_result is not None:
        # Cache hit
        entry, similarity = cache_result
        state.stats.hit_count += 1
        state.stats.total_entries = state.cache.total_entries

        logger.info(f"Cache HIT: '{query_text[:50]}...' matched '{entry.query_text[:50]}...' "
                     f"(sim={similarity:.4f})")

        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=entry.query_text,
            similarity_score=round(similarity, 4),
            results=entry.result,
            dominant_cluster=dominant_cluster,
        )

    # 4. Cache miss — query ChromaDB
    state.stats.miss_count += 1

    chroma_results = state.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=request.top_k,
        where={"label_index": {"$gte": 0}},  # no-op filter to ensure metadata is returned
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    documents = []
    if chroma_results and chroma_results['ids'] and chroma_results['ids'][0]:
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            doc_text = chroma_results['documents'][0][i] if chroma_results['documents'] else ""
            metadata = chroma_results['metadatas'][0][i] if chroma_results['metadatas'] else {}
            distance = chroma_results['distances'][0][i] if chroma_results['distances'] else 0.0

            documents.append(DocumentResult(
                doc_id=doc_id,
                text=doc_text[:500],  # Truncate for response
                label=metadata.get('original_label', 'unknown'),
                score=round(1.0 - distance, 4),  # Convert distance to similarity
            ))

    # 5. Insert into cache
    state.cache.insert(
        query_embedding=query_embedding,
        query_text=query_text,
        result=documents,
        cluster_memberships=memberships,
    )
    state.stats.total_entries = state.cache.total_entries

    logger.info(f"Cache MISS: '{query_text[:50]}...' → {len(documents)} results, "
                f"cluster={dominant_cluster}")

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        results=documents,
        dominant_cluster=dominant_cluster,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Return cache hit/miss statistics."""
    if state.cache is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    state.stats.total_entries = state.cache.total_entries

    return CacheStatsResponse(
        total_entries=state.stats.total_entries,
        hit_count=state.stats.hit_count,
        miss_count=state.stats.miss_count,
        hit_rate=round(state.stats.hit_rate, 4),
    )


@router.delete("/cache", response_model=CacheDeleteResponse)
async def delete_cache():
    """Flush all cache entries and reset statistics."""
    if state.cache is None:
        raise HTTPException(status_code=503, detail="Service not ready.")

    entries_removed = state.cache.flush()
    state.stats.reset()

    logger.info(f"Cache cleared: {entries_removed} entries removed")

    return CacheDeleteResponse(
        message="Cache cleared",
        entries_removed=entries_removed,
    )
