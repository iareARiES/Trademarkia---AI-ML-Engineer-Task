"""
Threshold analysis: sweep τ values, generate precision-recall data.
Usage: python -m src.cache.threshold_analysis
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_subject(text: str) -> str:
    """Extract subject line from newsgroup post."""
    import re
    match = re.search(r'^Subject:[ \t]*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
    if match:
        subject = match.group(1).strip()
        subject = re.sub(r'^(Re:\s*)+', '', subject, flags=re.IGNORECASE).strip()
        return subject if subject else None
    return None


def generate_paraphrase_pairs(n_pairs: int = 300, seed: int = 42) -> list:
    """
    Generate labelled (text_a, text_b, is_paraphrase) pairs.

    Strategy:
      - Positive pairs: subject lines from the same specific thread/topic within
        a category (high semantic similarity expected)
      - Negative pairs: subject lines from different categories (low similarity expected)
    """
    rng = np.random.RandomState(seed)
    dataset = fetch_20newsgroups(subset='all', remove=())

    # Extract subjects and group by label
    groups = defaultdict(list)
    for i, (text, label) in enumerate(zip(dataset.data, dataset.target)):
        subject = extract_subject(text)
        if subject and len(subject) > 10:
            groups[int(label)].append((i, subject))

    pairs = []
    labels_list = list(groups.keys())

    # Positive pairs: same category subjects (topically related queries)
    n_pos = n_pairs // 2
    for _ in range(n_pos * 3):  # oversample, filter later
        if len(pairs) >= n_pos:
            break
        label = rng.choice(labels_list)
        items = groups[label]
        if len(items) < 2:
            continue
        idx_a, idx_b = rng.choice(len(items), size=2, replace=False)
        subj_a = items[idx_a][1]
        subj_b = items[idx_b][1]
        # Skip if identical
        if subj_a.lower() == subj_b.lower():
            continue
        pairs.append((subj_a, subj_b, True))

    # Negative pairs: different category subjects
    n_neg = n_pairs // 2
    for _ in range(n_neg * 3):
        if len(pairs) >= n_pos + n_neg:
            break
        l1, l2 = rng.choice(labels_list, size=2, replace=False)
        items_a = groups[l1]
        items_b = groups[l2]
        if not items_a or not items_b:
            continue
        idx_a = rng.choice(len(items_a))
        idx_b = rng.choice(len(items_b))
        subj_a = items_a[idx_a][1]
        subj_b = items_b[idx_b][1]
        pairs.append((subj_a, subj_b, False))

    return pairs


def sweep_threshold(pairs: list, model: SentenceTransformer,
                    tau_values: np.ndarray) -> list:
    """Evaluate precision/recall at each τ value."""
    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    embeddings_a = model.encode(texts_a, normalize_embeddings=True)
    embeddings_b = model.encode(texts_b, normalize_embeddings=True)

    sims = np.array([float(np.dot(a, b)) for a, b in zip(embeddings_a, embeddings_b)])

    # Log similarity distributions
    pos_sims = [s for s, l in zip(sims, labels) if l]
    neg_sims = [s for s, l in zip(sims, labels) if not l]
    logger.info(f"Positive pair sims — mean: {np.mean(pos_sims):.4f}, "
                f"min: {np.min(pos_sims):.4f}, max: {np.max(pos_sims):.4f}")
    logger.info(f"Negative pair sims — mean: {np.mean(neg_sims):.4f}, "
                f"min: {np.min(neg_sims):.4f}, max: {np.max(neg_sims):.4f}")

    results = []
    for tau in tau_values:
        predictions = sims >= tau
        tp = sum(1 for pred, label in zip(predictions, labels) if pred and label)
        fp = sum(1 for pred, label in zip(predictions, labels) if pred and not label)
        fn = sum(1 for pred, label in zip(predictions, labels) if not pred and label)
        tn = sum(1 for pred, label in zip(predictions, labels) if not pred and not label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'tau': round(float(tau), 2),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })

    return results


def main():
    with open(PROJECT_ROOT / 'config.yaml') as f:
        config = yaml.safe_load(f)

    logger.info("Loading embedding model...")
    model = SentenceTransformer(config['model']['name'])

    logger.info("Generating paraphrase pairs from subject lines...")
    pairs = generate_paraphrase_pairs(n_pairs=300)
    n_pos = sum(1 for p in pairs if p[2])
    n_neg = sum(1 for p in pairs if not p[2])
    logger.info(f"Generated {len(pairs)} pairs ({n_pos} positive, {n_neg} negative)")

    tau_values = np.arange(0.10, 0.98, 0.02)
    logger.info(f"Sweeping τ over {len(tau_values)} values...")
    results = sweep_threshold(pairs, model, tau_values)

    # Find F1-optimal τ
    best = max(results, key=lambda r: r['f1'])
    logger.info(f"\nBest τ = {best['tau']:.2f} (F1={best['f1']:.4f}, "
                f"Precision={best['precision']:.4f}, Recall={best['recall']:.4f})")

    # Save results
    artefacts_dir = PROJECT_ROOT / config['clustering']['artefacts_dir']
    output = {
        'recommended_tau': best['tau'],
        'best_f1': best['f1'],
        'sweep_results': results,
        'n_pairs': len(pairs),
        'n_positive': n_pos,
        'n_negative': n_neg,
    }
    output_path = artefacts_dir / 'threshold_analysis.json'
    with open(str(output_path), 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print key part of the table
    logger.info("\n  τ     | Prec   | Recall | F1     | TP  | FP  | FN")
    logger.info("  " + "-" * 52)
    for r in results:
        if r['tau'] >= 0.30:
            marker = " ← BEST" if r['tau'] == best['tau'] else ""
            logger.info(f"  {r['tau']:.2f}  | {r['precision']:.4f} | {r['recall']:.4f} | "
                        f"{r['f1']:.4f} | {r['tp']:>3} | {r['fp']:>3} | {r['fn']:>3}{marker}")


if __name__ == '__main__':
    main()
