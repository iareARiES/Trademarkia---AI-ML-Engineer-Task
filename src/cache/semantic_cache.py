"""
Semantic Cache Layer — cluster-aware, in-memory cache for paraphrase detection.
Pure Python implementation with no external caching dependencies.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    """Single cache entry holding a query-result pair with metadata."""
    query_embedding: np.ndarray  # 384-dim, L2-normalised
    query_text: str
    result: Any
    cluster_memberships: np.ndarray  # c-dim soft membership vector
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class CacheStats:
    """Tracks cache hit/miss statistics."""
    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def reset(self):
        self.total_entries = 0
        self.hit_count = 0
        self.miss_count = 0


class SemanticCache:
    """
    Cluster-bucketed semantic cache.

    Structure: Dict[int, List[CacheEntry]]
    - Outer key: dominant_cluster_id
    - Inner list: cache entries belonging to that cluster bucket

    Lookup narrows search to relevant cluster buckets based on
    query's soft memberships, then performs cosine similarity scan.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        membership_threshold: float = 0.15,
        bucket_cap: int = 200,
        eviction_policy: str = 'fifo',
    ):
        self.similarity_threshold = similarity_threshold
        self.membership_threshold = membership_threshold
        self.bucket_cap = bucket_cap
        self.eviction_policy = eviction_policy
        self.cache: Dict[int, List[CacheEntry]] = {}

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_memberships: np.ndarray,
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Search cache for a semantically similar query.

        Args:
            query_embedding: L2-normalised 384-dim vector
            cluster_memberships: Soft membership vector (c-dim)

        Returns:
            (CacheEntry, similarity_score) if hit, None if miss.
        """
        # 1. Identify candidate buckets (clusters with membership > threshold)
        candidate_buckets = [
            k for k, membership in enumerate(cluster_memberships)
            if membership > self.membership_threshold
        ]

        if not candidate_buckets:
            # Fallback: use dominant cluster
            candidate_buckets = [int(np.argmax(cluster_memberships))]

        # 2. Gather entries from candidate buckets
        best_entry = None
        best_sim = -1.0

        for bucket_id in candidate_buckets:
            entries = self.cache.get(bucket_id, [])
            for entry in entries:
                # 3. Cosine similarity (dot product since both are L2-normalised)
                sim = float(np.dot(query_embedding, entry.query_embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        # 4. Return hit if similarity >= threshold
        if best_entry is not None and best_sim >= self.similarity_threshold:
            best_entry.hit_count += 1
            return best_entry, best_sim

        return None

    def insert(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        result: Any,
        cluster_memberships: np.ndarray,
    ) -> CacheEntry:
        """
        Insert a new entry into the cache.

        Args:
            query_embedding: L2-normalised 384-dim vector
            query_text: Original query string
            result: Search result payload
            cluster_memberships: Soft membership vector (c-dim)

        Returns:
            The newly created CacheEntry.
        """
        # Determine dominant cluster
        dominant_cluster = int(np.argmax(cluster_memberships))

        entry = CacheEntry(
            query_embedding=query_embedding,
            query_text=query_text,
            result=result,
            cluster_memberships=cluster_memberships,
        )

        # Initialise bucket if needed
        if dominant_cluster not in self.cache:
            self.cache[dominant_cluster] = []

        self.cache[dominant_cluster].append(entry)

        # Enforce per-bucket cap
        bucket = self.cache[dominant_cluster]
        if len(bucket) > self.bucket_cap:
            if self.eviction_policy == 'lru':
                # Evict least recently hit (lowest hit_count, then oldest)
                bucket.sort(key=lambda e: (e.hit_count, e.timestamp))
                self.cache[dominant_cluster] = bucket[1:]
            else:
                # FIFO: evict oldest
                self.cache[dominant_cluster] = bucket[1:]

        return entry

    def flush(self) -> int:
        """Clear all cache entries. Returns number of entries removed."""
        total = self.total_entries
        self.cache.clear()
        return total

    @property
    def total_entries(self) -> int:
        """Total number of entries across all buckets."""
        return sum(len(entries) for entries in self.cache.values())

    def bucket_stats(self) -> Dict[int, int]:
        """Return entry count per bucket."""
        return {k: len(v) for k, v in self.cache.items()}
