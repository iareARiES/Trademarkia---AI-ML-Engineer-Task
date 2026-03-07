"""
Cluster analysis and validation helpers.
Provides tools for inspecting FCM clustering results.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def top_terms_per_cluster(
    documents: List[str],
    membership_matrix: np.ndarray,
    n_terms: int = 10,
    threshold: float = 0.3,
) -> Dict[int, List[str]]:
    """
    Compute top TF-IDF terms for each cluster.
    Uses documents where membership u_ik > threshold as dominant members.
    """
    n_clusters = membership_matrix.shape[1]
    result = {}

    for k in range(n_clusters):
        # Get documents with strong membership in this cluster
        mask = membership_matrix[:, k] > threshold
        cluster_docs = [doc for doc, m in zip(documents, mask) if m]

        if not cluster_docs:
            result[k] = []
            continue

        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(cluster_docs)
        feature_names = tfidf.get_feature_names_out()

        # Average TF-IDF scores across cluster documents
        avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[-n_terms:][::-1]
        result[k] = [feature_names[i] for i in top_indices]

    return result


def get_representative_docs(
    membership_matrix: np.ndarray,
    documents: List[str],
    n_docs: int = 3,
) -> Dict[int, List[Tuple[int, float, str]]]:
    """
    For each cluster, surface the n_docs with highest membership.
    Returns dict mapping cluster_id -> [(doc_index, membership, doc_text_preview), ...].
    """
    n_clusters = membership_matrix.shape[1]
    result = {}

    for k in range(n_clusters):
        memberships = membership_matrix[:, k]
        top_indices = memberships.argsort()[-n_docs:][::-1]
        result[k] = [
            (int(idx), float(memberships[idx]), documents[idx][:200])
            for idx in top_indices
        ]

    return result


def get_boundary_docs(
    membership_matrix: np.ndarray,
    documents: List[str],
    max_membership_threshold: float = 0.4,
    n_docs: int = 10,
) -> List[Tuple[int, float, str]]:
    """
    Surface documents where max membership < threshold — genuinely ambiguous docs.
    """
    max_memberships = membership_matrix.max(axis=1)
    boundary_mask = max_memberships < max_membership_threshold
    boundary_indices = np.where(boundary_mask)[0]

    # Sort by max membership (most ambiguous first)
    sorted_indices = boundary_indices[max_memberships[boundary_indices].argsort()]

    return [
        (int(idx), float(max_memberships[idx]), documents[idx][:200])
        for idx in sorted_indices[:n_docs]
    ]


def cross_label_heatmap_data(
    membership_matrix: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
) -> np.ndarray:
    """
    Compute heatmap data: rows=original 20 labels, cols=discovered clusters.
    Values = mean membership for documents with that label.
    """
    n_labels = len(label_names)
    n_clusters = membership_matrix.shape[1]
    heatmap = np.zeros((n_labels, n_clusters))

    for label_idx in range(n_labels):
        mask = labels == label_idx
        if mask.sum() > 0:
            heatmap[label_idx] = membership_matrix[mask].mean(axis=0)

    return heatmap


def membership_entropy(membership_matrix: np.ndarray) -> np.ndarray:
    """Per-document entropy: H = -Σ_k u_ik log(u_ik)."""
    u_safe = np.clip(membership_matrix, 1e-10, 1.0)
    return -np.sum(u_safe * np.log(u_safe), axis=1)
