"""
Fuzzy C-Means clustering on document embeddings.
PCA reduction → KMeans-seeded FCM → cluster sweep → persistence.

Uses a custom FCM implementation because skfuzzy 0.5.0 has a known issue
producing uniform memberships on high-dimensional embedding data.

Usage: python -m src.clustering.fuzzy_cluster
"""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / 'config.yaml') as f:
        return yaml.safe_load(f)


def fuzzy_cmeans(data: np.ndarray, c: int, m: float = 2.0,
                 error: float = 0.005, maxiter: int = 300,
                 seed: int = 42) -> tuple:
    """
    Custom Fuzzy C-Means implementation.
    Initialises centroids from KMeans to avoid degenerate uniform solutions.

    Args:
        data: (n_samples, n_features)
        c: number of clusters
        m: fuzzifier exponent (>1, typically 2.0)
        error: convergence tolerance on membership change
        maxiter: maximum iterations

    Returns:
        (centers, membership_matrix, objective_history, fpc)
        - centers: (c, n_features)
        - membership_matrix: (n_samples, c), rows sum to 1
        - objective_history: list of objective values
        - fpc: fuzzy partition coefficient
    """
    n, d = data.shape
    rng = np.random.RandomState(seed)

    # Initialize centroids from KMeans
    km = KMeans(n_clusters=c, random_state=seed, n_init=3, max_iter=50)
    km.fit(data)
    centers = km.cluster_centers_.copy()  # (c, d)

    # Compute initial membership matrix
    U = _compute_memberships(data, centers, m)

    obj_history = []

    for iteration in range(maxiter):
        # Update centers
        centers_new = _update_centers(data, U, m)

        # Update memberships
        U_new = _compute_memberships(data, centers_new, m)

        # Compute objective
        obj = _compute_objective(data, centers_new, U_new, m)
        obj_history.append(obj)

        # Check convergence
        delta = np.abs(U_new - U).max()
        centers = centers_new
        U = U_new

        if delta < error:
            logger.debug(f"  Converged at iteration {iteration + 1} (delta={delta:.6f})")
            break

    # Fuzzy Partition Coefficient
    fpc = float(np.mean(np.sum(U ** 2, axis=1)))

    return centers, U, obj_history, fpc


def _compute_memberships(data: np.ndarray, centers: np.ndarray,
                         m: float) -> np.ndarray:
    """Compute membership matrix U (n_samples, c)."""
    n = data.shape[0]
    c = centers.shape[0]
    exponent = 2.0 / (m - 1.0)

    # Distances: (n, c)
    dists = np.zeros((n, c))
    for k in range(c):
        diff = data - centers[k]
        dists[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))

    # Avoid division by zero
    dists = np.clip(dists, 1e-10, None)

    # Membership formula: u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
    U = np.zeros((n, c))
    for k in range(c):
        ratio_sum = np.zeros(n)
        for j in range(c):
            ratio_sum += (dists[:, k] / dists[:, j]) ** exponent
        U[:, k] = 1.0 / ratio_sum

    return U


def _update_centers(data: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """Update cluster centers as weighted means."""
    c = U.shape[1]
    d = data.shape[1]
    centers = np.zeros((c, d))

    for k in range(c):
        weights = U[:, k] ** m  # (n,)
        centers[k] = np.average(data, axis=0, weights=weights)

    return centers


def _compute_objective(data: np.ndarray, centers: np.ndarray,
                       U: np.ndarray, m: float) -> float:
    """Compute FCM objective function J = sum_i sum_k u_ik^m * d_ik^2."""
    obj = 0.0
    c = centers.shape[0]
    for k in range(c):
        diff = data - centers[k]
        dists_sq = np.sum(diff ** 2, axis=1)
        obj += np.sum((U[:, k] ** m) * dists_sq)
    return float(obj)


def compute_partition_coefficient(U: np.ndarray) -> float:
    """PC = (1/n) * sum(u_ik^2). Range [1/c, 1]. Higher = crisper."""
    return float(np.mean(np.sum(U ** 2, axis=1)))


def compute_xie_beni(U: np.ndarray, centers: np.ndarray,
                     data: np.ndarray, m: float = 2.0) -> float:
    """Xie-Beni index: compactness / separation. Lower = better."""
    n = data.shape[0]
    c = centers.shape[0]

    compactness = 0.0
    for k in range(c):
        diff = data - centers[k]
        dists_sq = np.sum(diff ** 2, axis=1)
        compactness += np.sum((U[:, k] ** m) * dists_sq)

    min_sep = float('inf')
    for i in range(c):
        for j in range(i + 1, c):
            d = np.sum((centers[i] - centers[j]) ** 2)
            if d < min_sep:
                min_sep = d

    if min_sep == 0:
        return float('inf')

    return float(compactness / (n * min_sep))


def compute_avg_entropy(U: np.ndarray) -> float:
    """Average per-document entropy. Lower = crisper assignments."""
    u_safe = np.clip(U, 1e-10, 1.0)
    doc_entropy = -np.sum(u_safe * np.log(u_safe), axis=1)
    return float(np.mean(doc_entropy))


def main():
    config = load_config()
    clustering_cfg = config['clustering']
    artefacts_dir = PROJECT_ROOT / clustering_cfg['artefacts_dir']
    os.makedirs(artefacts_dir, exist_ok=True)

    t_start = time.time()

    # --- 1. Load embeddings ---
    embeddings_path = artefacts_dir / 'embeddings.npy'
    if not embeddings_path.exists():
        logger.error("Embeddings not found! Run ingestion first: python -m src.ingestion.ingest")
        return

    embeddings = np.load(str(embeddings_path))
    logger.info(f"Loaded embeddings: shape {embeddings.shape}")

    # --- 2. PCA dimensionality reduction ---
    pca_dims = clustering_cfg['pca_dims']
    logger.info(f"Applying PCA: {embeddings.shape[1]} → {pca_dims} dimensions...")
    pca = PCA(n_components=pca_dims, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    variance_retained = sum(pca.explained_variance_ratio_)
    logger.info(f"PCA variance retained: {variance_retained:.4f} ({variance_retained * 100:.1f}%)")

    # --- 3. Cluster sweep ---
    c_values = clustering_cfg['n_clusters_sweep']
    m = clustering_cfg['fuzzifier']
    error_tol = clustering_cfg['error']
    maxiter = clustering_cfg['maxiter']
    n_restarts = clustering_cfg['n_restarts']

    sweep_results = []

    for c in c_values:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FCM with c={c} (m={m}, {n_restarts} restarts)...")

        best_obj = float('inf')
        best_result = None

        for restart in range(n_restarts):
            seed = 42 + restart * 7
            centers, U, obj_hist, fpc = fuzzy_cmeans(
                embeddings_pca, c=c, m=m, error=error_tol,
                maxiter=maxiter, seed=seed,
            )
            final_obj = obj_hist[-1] if obj_hist else float('inf')

            if final_obj < best_obj:
                best_obj = final_obj
                best_result = (centers, U, obj_hist, fpc)
                logger.info(f"  Restart {restart + 1}: obj={final_obj:.2f}, fpc={fpc:.4f} (new best)")
            else:
                logger.info(f"  Restart {restart + 1}: obj={final_obj:.2f}, fpc={fpc:.4f}")

        centers, U, obj_hist, fpc = best_result
        pc = compute_partition_coefficient(U)
        xb = compute_xie_beni(U, centers, embeddings_pca, m)
        avg_ent = compute_avg_entropy(U)

        result = {
            'c': c,
            'pc': round(pc, 6),
            'xb': round(xb, 4),
            'avg_entropy': round(avg_ent, 4),
            'fpc': round(fpc, 6),
            'final_objective': round(best_obj, 2),
        }
        sweep_results.append(result)
        logger.info(f"  c={c}: PC={pc:.4f}, XB={xb:.4f}, Entropy={avg_ent:.4f}")

    # --- 4. Select best c ---
    # Primary: maximize PC. If tie, minimize XB.
    best = max(sweep_results, key=lambda r: (r['pc'], -r['xb']))
    best_c = best['c']
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Best c={best_c} (PC={best['pc']:.4f}, XB={best['xb']:.4f})")

    # --- 5. Final FCM with best c ---
    logger.info(f"\nFinal FCM with c={best_c}...")
    best_obj = float('inf')
    best_final = None
    for restart in range(n_restarts):
        seed = 42 + restart * 7
        centers, U, obj_hist, fpc = fuzzy_cmeans(
            embeddings_pca, c=best_c, m=m, error=error_tol,
            maxiter=maxiter, seed=seed,
        )
        final_obj = obj_hist[-1] if obj_hist else float('inf')
        if final_obj < best_obj:
            best_obj = final_obj
            best_final = (centers, U, fpc)
            logger.info(f"  Restart {restart + 1}: obj={final_obj:.2f}, fpc={fpc:.4f} (new best)")
        else:
            logger.info(f"  Restart {restart + 1}: obj={final_obj:.2f}, fpc={fpc:.4f}")

    centers, U, fpc = best_final
    t_cluster = time.time()

    # --- 6. Membership distribution summary ---
    max_u = U.max(axis=1)
    logger.info(f"\nMembership distribution:")
    logger.info(f"  Max membership — mean: {max_u.mean():.4f}, median: {np.median(max_u):.4f}")
    logger.info(f"  Max membership — min: {max_u.min():.4f}, max: {max_u.max():.4f}")
    for thresh in [0.3, 0.4, 0.5, 0.7, 0.9]:
        count = (max_u > thresh).sum()
        logger.info(f"  Docs with max_u > {thresh}: {count} ({count / len(U) * 100:.1f}%)")

    # --- 7. Persist artefacts ---
    logger.info("\nSaving clustering artefacts...")
    np.save(str(artefacts_dir / 'membership_matrix.npy'), U.astype(np.float32))
    np.save(str(artefacts_dir / 'cluster_centers.npy'), centers.astype(np.float32))
    joblib.dump(pca, str(artefacts_dir / 'pca_model.pkl'))

    doc_id_path = artefacts_dir / 'doc_id_order.json'
    doc_id_order = []
    if doc_id_path.exists():
        with open(doc_id_path) as f:
            doc_id_order = json.load(f)

    fcm_meta = {
        'c': best_c,
        'm': m,
        'pc_score': best['pc'],
        'xb_score': best['xb'],
        'avg_entropy': best['avg_entropy'],
        'pca_dims': pca_dims,
        'pca_variance_retained': float(variance_retained),
        'n_documents': int(embeddings.shape[0]),
        'sweep_results': sweep_results,
        'doc_id_order': doc_id_order,
        'timing_seconds': round(t_cluster - t_start, 2),
    }
    with open(str(artefacts_dir / 'fcm_meta.json'), 'w') as f:
        json.dump(fcm_meta, f, indent=2)

    logger.info(f"Clustering complete in {fcm_meta['timing_seconds']}s")
    logger.info(f"  Membership matrix: {U.shape}")
    logger.info(f"  Cluster centers: {centers.shape}")
    logger.info(f"  Artefacts saved to: {artefacts_dir}")


if __name__ == '__main__':
    main()
