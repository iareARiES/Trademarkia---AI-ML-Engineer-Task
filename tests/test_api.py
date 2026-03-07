"""
Integration tests for the FastAPI API.
Uses FastAPI TestClient — does NOT require running uvicorn.
NOTE: These tests require artefacts to have been generated first
      (run ingestion + clustering pipelines before running these tests).
"""

import pytest

# These tests only run if artefacts exist
try:
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    HAS_ARTEFACTS = True
except Exception:
    HAS_ARTEFACTS = False
    client = None


@pytest.mark.skipif(not HAS_ARTEFACTS, reason="Artefacts not generated yet")
class TestQueryEndpoint:
    def test_query_returns_results(self):
        response = client.post(
            "/query",
            json={"query": "What are the best graphics cards for gaming?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "cache_hit" in data
        assert "results" in data
        assert "dominant_cluster" in data

    def test_duplicate_query_hits_cache(self):
        query = "Tell me about space exploration and NASA missions"

        # First query — should be a miss
        r1 = client.post("/query", json={"query": query})
        assert r1.status_code == 200
        assert r1.json()["cache_hit"] is False

        # Same query — should hit cache
        r2 = client.post("/query", json={"query": query})
        assert r2.status_code == 200
        assert r2.json()["cache_hit"] is True

    def test_query_validation_short(self):
        response = client.post("/query", json={"query": "ab"})
        assert response.status_code == 422  # Validation error

    def test_query_with_top_k(self):
        response = client.post(
            "/query",
            json={"query": "Discussion about encryption and privacy", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        if not data["cache_hit"]:
            assert len(data["results"]) <= 3


@pytest.mark.skipif(not HAS_ARTEFACTS, reason="Artefacts not generated yet")
class TestCacheStatsEndpoint:
    def test_cache_stats(self):
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "hit_count" in data
        assert "miss_count" in data
        assert "hit_rate" in data


@pytest.mark.skipif(not HAS_ARTEFACTS, reason="Artefacts not generated yet")
class TestCacheDeleteEndpoint:
    def test_delete_cache(self):
        # Insert something first
        client.post(
            "/query",
            json={"query": "Temporary query for delete test clearing"},
        )

        response = client.delete("/cache")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Cache cleared"
        assert "entries_removed" in data

        # Verify cache is empty
        stats = client.get("/cache/stats").json()
        assert stats["total_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0


@pytest.mark.skipif(not HAS_ARTEFACTS, reason="Artefacts not generated yet")
class TestHealthEndpoint:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
