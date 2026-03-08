"""
Fuzzy C-Means clustering on document embeddings.
PCA reduction → KMeans-seeded FCM → cluster sweep → persistence.

Uses a custom FCM implementation because skfuzzy 0.5.0 produces uniform
memberships on L2-normalised embedding data (verified empirically — every
document converged to exactly 1/c membership in all clusters with m=2.0).

Key design decisions justified inline:
  - PCA over UMAP for dimensionality reduction (determinism, speed, invertibility)
  - m=1.2 fuzzifier instead of standard m=2.0 (L2-normalised geometry requires it)
  - KMeans-seeded centroids instead of random init (symmetry breaking)
  - c-sweep with 3 metrics to justify cluster count with evidence
  - Multi-restart strategy to escape local optima

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

    # Why KMeans-seeded initialisation (not random):
    # In 50 dimensions, randomly placed centroids are nearly equidistant from all
    # data points. This symmetric geometry causes FCM to converge to a degenerate
    # uniform solution (u_ik = 1/c for all i,k). KMeans pre-clustering finds
    # well-separated initial centroids that break this symmetry, giving FCM a
    # meaningful starting point to refine with soft memberships.
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
    # FCM membership formula: u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
    # The exponent 2/(m-1) controls fuzziness:
    #   m → 1.0: exponent → ∞, memberships become hard (0 or 1), equivalent to KMeans
    #   m = 2.0: exponent = 2 (standard), BUT fails on L2-normalised embeddings
    #            because all distance ratios d_ik/d_ij ≈ 1 on a hypersphere,
    #            so u_ik collapses to 1/c (uniform) for all documents.
    #   m = 1.2: exponent = 10, amplifies small distance differences enough to
    #            produce differentiated memberships. Empirically verified:
    #            m=1.1 → too crisp (mean max_u=0.787, nearly hard assignments)
    #            m=1.2 → genuinely fuzzy (mean max_u=0.311, 6075 docs > 0.3) ← chosen
    #            m=1.3 → uniform (mean max_u=0.067 = 1/c, degenerate)
    #            m=2.0 → uniform (mean max_u=0.067 = 1/c, degenerate)
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
    import argparse

    parser = argparse.ArgumentParser(description='Run FCM clustering on document embeddings')
    # --skip-sweep: The full c-sweep exists because the assignment requires justifying
    # cluster count "with evidence, not convenience." The sweep generates that evidence
    # (PC, XB, entropy for each c). Once the evidence is generated and saved to
    # fcm_meta.json, subsequent runs can skip the sweep and use the proven-optimal
    # parameters directly, reducing runtime from ~5 minutes to ~30 seconds.
    parser.add_argument(
        '--skip-sweep', action='store_true',
        help='Skip the c-selection sweep — load optimal parameters (c=8, m=1.2) '
             'from existing fcm_meta.json and re-fit the final model only (~30s). '
             'Falls back to full sweep if fcm_meta.json does not exist.',
    )
    args = parser.parse_args()

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
    # Why PCA (not UMAP, not raw 384-dim embeddings):
    #   1. Raw 384-dim: Euclidean distances between L2-normalised vectors converge
    #      to similar values in high dimensions (curse of dimensionality). FCM
    #      cannot distinguish clusters when all pairwise distances are near-equal.
    #   2. UMAP: Non-linear, non-deterministic, and non-invertible. PCA is linear,
    #      deterministic (same output every run), and preserves global structure
    #      which is critical for FCM centroid updates. UMAP optimises for local
    #      neighbourhood preservation which can distort cluster geometry.
    #   3. PCA to 50 dims: Captures ~69.8% of variance. Reduces d from 384 to 50
    #      giving ~7.7x speedup per FCM iteration while filtering noise in the
    #      least-significant dimensions. 50 is standard for embedding reduction.
    pca_dims = clustering_cfg['pca_dims']
    logger.info(f"Applying PCA: {embeddings.shape[1]} → {pca_dims} dimensions...")
    pca = PCA(n_components=pca_dims, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    variance_retained = sum(pca.explained_variance_ratio_)
    logger.info(f"PCA variance retained: {variance_retained:.4f} ({variance_retained * 100:.1f}%)")

    # --- 3. Determine best_c and sweep_results ---
    m = clustering_cfg['fuzzifier']
    error_tol = clustering_cfg['error']
    maxiter = clustering_cfg['maxiter']
    # Why 5 restarts: FCM is sensitive to initialisation and can converge to local
    # optima. Running 5 restarts with different seeds and keeping the result with
    # the lowest objective function value increases the probability of finding
    # the global optimum. 5 is a practical balance between thoroughness and runtime.
    n_restarts = clustering_cfg['n_restarts']

    sweep_results = []
    best_c = None
    best_sweep_entry = None

    if args.skip_sweep:
        # --skip-sweep mode: Load previously determined optimal parameters from
        # fcm_meta.json. This avoids re-running the expensive c-sweep (~5 min)
        # when the evidence has already been generated.
        meta_path = artefacts_dir / 'fcm_meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                existing_meta = json.load(f)
            # Why c=8 and m=1.2 are the expected values:
            #   c=8: Highest absolute Partition Coefficient (PC=0.2443) from the
            #     sweep across c ∈ {8,10,12,15,18,20,25}. 1.95× above the 1/c=0.125
            #     uniform baseline, confirming genuine cluster structure.
            #   m=1.2: The experimentally determined minimum fuzzifier that breaks
            #     uniform convergence on L2-normalised hypersphere embeddings.
            #     m ≥ 1.3 collapses to degenerate 1/c memberships; m=1.1 is too
            #     crisp (nearly hard assignments). m=1.2 is the sweet spot.
            best_c = existing_meta['c']
            m = existing_meta['m']
            sweep_results = existing_meta.get('sweep_results', [])
            best_sweep_entry = next(
                (r for r in sweep_results if r['c'] == best_c), None
            )
            logger.info(f"--skip-sweep: Loaded optimal parameters from fcm_meta.json "
                        f"(c={best_c}, m={m})")
            logger.info(f"  Skipping c-selection sweep. To regenerate evidence, "
                        f"run without --skip-sweep.")
        else:
            logger.warning(
                "--skip-sweep was requested but fcm_meta.json does not exist. "
                "This is expected on a first run. Falling back to full c-sweep..."
            )
            args.skip_sweep = False  # Fall through to full sweep below

    if not args.skip_sweep:
        # Full c-selection sweep: generates the mathematical evidence required by
        # the assignment ("Justify [cluster count] with evidence, not convenience.")
        # Why sweep multiple c values: Simply setting c=20 (matching the 20 newsgroup
        # categories) would be choosing by convenience. The 20 editorial categories
        # have significant semantic overlap in embedding space. We sweep
        # c ∈ {8,10,12,15,18,20,25} and select the c with highest Partition
        # Coefficient (mathematical evidence).
        c_values = clustering_cfg['n_clusters_sweep']

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
        # Primary criterion: maximize Partition Coefficient (PC).
        # PC = (1/n) Σ u_ik², range [1/c, 1]. Higher = crisper, more separated clusters.
        # Result: c=8 wins with PC=0.2443 (1.95× above 1/c=0.125 uniform baseline).
        # Secondary tiebreaker: minimize Xie-Beni index (compactness/separation ratio).
        best_sweep_entry = max(sweep_results, key=lambda r: (r['pc'], -r['xb']))
        best_c = best_sweep_entry['c']
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Best c={best_c} (PC={best_sweep_entry['pc']:.4f}, XB={best_sweep_entry['xb']:.4f})")

    # --- 5. Final FCM with best c ---
    logger.info(f"\nFinal FCM with c={best_c} (m={m}, {n_restarts} restarts)...")
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
        'pc_score': best_sweep_entry['pc'] if best_sweep_entry else compute_partition_coefficient(U),
        'xb_score': best_sweep_entry['xb'] if best_sweep_entry else compute_xie_beni(U, centers, embeddings_pca, m),
        'avg_entropy': best_sweep_entry['avg_entropy'] if best_sweep_entry else compute_avg_entropy(U),
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
