"""
Unit tests for the SemanticCache.
"""

import numpy as np
import pytest

from src.cache.semantic_cache import CacheEntry, CacheStats, SemanticCache


def make_embedding(dim: int = 384, seed: int = 0) -> np.ndarray:
    """Create a deterministic, L2-normalised random embedding."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def make_memberships(n_clusters: int = 15, dominant: int = 0) -> np.ndarray:
    """Create a membership vector with one dominant cluster."""
    memberships = np.full(n_clusters, 0.02, dtype=np.float32)
    memberships[dominant] = 0.7
    memberships /= memberships.sum()  # Normalise to sum=1
    return memberships


class TestCacheStats:
    def test_initial_state(self):
        stats = CacheStats()
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hit_count=3, miss_count=7)
        assert abs(stats.hit_rate - 0.3) < 1e-6

    def test_reset(self):
        stats = CacheStats(total_entries=10, hit_count=5, miss_count=5)
        stats.reset()
        assert stats.total_entries == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0


class TestSemanticCache:
    def setup_method(self):
        self.cache = SemanticCache(
            similarity_threshold=0.82,
            membership_threshold=0.15,
            bucket_cap=5,
            eviction_policy='fifo',
        )

    def test_empty_cache_returns_none(self):
        emb = make_embedding(seed=0)
        mem = make_memberships(dominant=0)
        result = self.cache.lookup(emb, mem)
        assert result is None

    def test_insert_and_hit(self):
        emb = make_embedding(seed=42)
        mem = make_memberships(dominant=3)

        self.cache.insert(
            query_embedding=emb,
            query_text="test query",
            result="test result",
            cluster_memberships=mem,
        )

        # Same embedding should be a hit (cosine sim = 1.0)
        result = self.cache.lookup(emb, mem)
        assert result is not None
        entry, sim = result
        assert sim >= 0.99  # Should be ~1.0
        assert entry.query_text == "test query"

    def test_different_embedding_is_miss(self):
        emb1 = make_embedding(seed=42)
        emb2 = make_embedding(seed=999)  # Very different
        mem = make_memberships(dominant=3)

        self.cache.insert(
            query_embedding=emb1,
            query_text="query 1",
            result="result 1",
            cluster_memberships=mem,
        )

        result = self.cache.lookup(emb2, mem)
        # Should be None (random embeddings have low cosine similarity)
        assert result is None

    def test_eviction_at_bucket_cap(self):
        mem = make_memberships(dominant=0)

        # Insert more than bucket_cap entries
        for i in range(7):
            emb = make_embedding(seed=i + 100)
            self.cache.insert(
                query_embedding=emb,
                query_text=f"query {i}",
                result=f"result {i}",
                cluster_memberships=mem,
            )

        # Bucket should be capped at bucket_cap (5)
        assert len(self.cache.cache[0]) <= self.cache.bucket_cap

    def test_flush_clears_all(self):
        mem = make_memberships(dominant=0)
        for i in range(3):
            emb = make_embedding(seed=i + 200)
            self.cache.insert(
                query_embedding=emb,
                query_text=f"q{i}",
                result=f"r{i}",
                cluster_memberships=mem,
            )

        assert self.cache.total_entries == 3
        removed = self.cache.flush()
        assert removed == 3
        assert self.cache.total_entries == 0

    def test_cross_cluster_isolation(self):
        emb1 = make_embedding(seed=10)
        emb2 = make_embedding(seed=20)
        mem_a = make_memberships(dominant=0)
        mem_b = make_memberships(dominant=5)

        self.cache.insert(emb1, "cluster 0 query", "r1", mem_a)
        self.cache.insert(emb2, "cluster 5 query", "r2", mem_b)

        assert 0 in self.cache.cache
        assert 5 in self.cache.cache
        assert len(self.cache.cache[0]) == 1
        assert len(self.cache.cache[5]) == 1

    def test_bucket_stats(self):
        for i in range(3):
            emb = make_embedding(seed=i + 300)
            mem = make_memberships(dominant=2)
            self.cache.insert(emb, f"q{i}", f"r{i}", mem)

        stats = self.cache.bucket_stats()
        assert stats[2] == 3


class TestCacheEntry:
    def test_defaults(self):
        emb = make_embedding()
        mem = make_memberships()
        entry = CacheEntry(
            query_embedding=emb,
            query_text="test",
            result="result",
            cluster_memberships=mem,
        )
        assert entry.hit_count == 0
        assert entry.timestamp > 0
