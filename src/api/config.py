"""
Configuration loader — reads config.yaml into typed Pydantic models.
"""

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


class ChromaConfig(BaseModel):
    persist_directory: str = './artefacts/chroma_db'
    collection_name: str = 'newsgroups_v1'


class ModelConfig(BaseModel):
    name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    batch_size: int = 64


class ClusteringConfig(BaseModel):
    artefacts_dir: str = './artefacts/clustering'
    membership_threshold: float = 0.15
    n_clusters_sweep: List[int] = [8, 10, 12, 15, 18, 20, 25]
    fuzzifier: float = 2.0
    error: float = 0.005
    maxiter: int = 300
    n_restarts: int = 5
    pca_dims: int = 50


class CacheConfig(BaseModel):
    similarity_threshold: float = 0.82
    bucket_cap: int = 200
    eviction_policy: str = 'fifo'


class ApiConfig(BaseModel):
    host: str = '0.0.0.0'
    port: int = 8000
    log_level: str = 'info'


class AppConfig(BaseModel):
    chroma: ChromaConfig = ChromaConfig()
    model: ModelConfig = ModelConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    cache: CacheConfig = CacheConfig()
    api: ApiConfig = ApiConfig()


def load_config(config_path: str = None) -> AppConfig:
    """Load config from YAML file, falling back to defaults."""
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent.parent / 'config.yaml')

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)
    except FileNotFoundError:
        return AppConfig()
