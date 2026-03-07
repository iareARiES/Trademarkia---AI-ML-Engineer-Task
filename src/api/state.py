"""
Singleton state management for the API.
Holds references to all shared resources: cache, stats, models, data.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from src.api.config import AppConfig, load_config
from src.cache.semantic_cache import CacheStats, SemanticCache

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Holds all application state as a single injectable object."""
    config: Optional[AppConfig] = None
    model: Optional[SentenceTransformer] = None
    collection: Optional[chromadb.Collection] = None
    pca_model: Optional[object] = None
    cluster_centers: Optional[np.ndarray] = None
    fcm_meta: Optional[dict] = None
    cache: Optional[SemanticCache] = None
    stats: CacheStats = field(default_factory=CacheStats)


# Module-level singleton
state = AppState()


def compute_memberships(
    query_embedding_pca: np.ndarray,
    cluster_centers: np.ndarray,
    m: float = 2.0,
) -> np.ndarray:
    """
    Compute soft FCM memberships for a query in PCA space.

    Uses the FCM membership formula:
    u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))

    Args:
        query_embedding_pca: PCA-projected query embedding (pca_dims,)
        cluster_centers: Cluster centroids in PCA space (c, pca_dims)
        m: Fuzzifier exponent

    Returns:
        Membership vector (c,) summing to 1.
    """
    c = cluster_centers.shape[0]
    exponent = 2.0 / (m - 1.0)

    # Distances from query to each centroid
    dists = np.linalg.norm(cluster_centers - query_embedding_pca, axis=1)
    dists = np.clip(dists, 1e-10, None)  # Avoid division by zero

    memberships = np.zeros(c)
    for i in range(c):
        ratio_sum = sum((dists[i] / dists[j]) ** exponent for j in range(c))
        memberships[i] = 1.0 / ratio_sum

    return memberships


def initialise_state():
    """Load all models and artefacts. Called during app startup."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(str(project_root / 'config.yaml'))
    state.config = config

    logger.info("Loading embedding model...")
    state.model = SentenceTransformer(config.model.name)

    logger.info("Connecting to ChromaDB...")
    persist_dir = str(project_root / config.chroma.persist_directory)
    client = chromadb.PersistentClient(path=persist_dir)
    state.collection = client.get_collection(name=config.chroma.collection_name)
    logger.info(f"ChromaDB collection '{config.chroma.collection_name}' loaded with {state.collection.count()} documents")

    logger.info("Loading clustering artefacts...")
    clustering_dir = project_root / config.clustering.artefacts_dir
    state.pca_model = joblib.load(str(clustering_dir / 'pca_model.pkl'))
    state.cluster_centers = np.load(str(clustering_dir / 'cluster_centers.npy'))

    with open(str(clustering_dir / 'fcm_meta.json')) as f:
        state.fcm_meta = json.load(f)

    logger.info(f"Loaded {state.fcm_meta['c']} cluster centers")

    logger.info("Initialising semantic cache...")
    state.cache = SemanticCache(
        similarity_threshold=config.cache.similarity_threshold,
        membership_threshold=config.clustering.membership_threshold,
        bucket_cap=config.cache.bucket_cap,
        eviction_policy=config.cache.eviction_policy,
    )
    state.stats = CacheStats()

    # Warm up HNSW index
    logger.info("Warming up HNSW index...")
    try:
        warmup_results = state.collection.query(
            query_embeddings=[np.zeros(384).tolist()],
            n_results=1,
        )
        logger.info("HNSW index warmed up successfully")
    except Exception as e:
        logger.warning(f"HNSW warmup failed (non-critical): {e}")

    logger.info("Application state initialised successfully!")
