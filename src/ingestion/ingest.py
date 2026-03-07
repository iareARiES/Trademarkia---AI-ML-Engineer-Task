"""
Ingestion pipeline: load 20 Newsgroups → preprocess → embed → upsert to ChromaDB.
Usage: python -m src.ingestion.ingest [--force]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

from src.ingestion.preprocessing import preprocess_document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve project root (support running as module or script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config() -> dict:
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_doc_id(text: str, index: int) -> str:
    """Generate a deterministic document ID from content hash."""
    content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
    return f"doc_{index:05d}_{content_hash}"


def main():
    parser = argparse.ArgumentParser(description='Ingest 20 Newsgroups into ChromaDB')
    parser.add_argument('--force', action='store_true', help='Force re-index (delete existing collection)')
    args = parser.parse_args()

    config = load_config()
    t_start = time.time()

    # --- 1. Load raw dataset ---
    logger.info("Loading 20 Newsgroups dataset via scikit-learn...")
    dataset = fetch_20newsgroups(
        subset='all',
        remove=(),
        data_home=str(PROJECT_ROOT / 'data')
    )
    raw_docs = dataset.data
    labels = dataset.target
    label_names = dataset.target_names
    logger.info(f"Loaded {len(raw_docs)} raw documents across {len(label_names)} categories")

    # --- 2. Preprocess ---
    logger.info("Preprocessing documents...")
    cleaned_docs = []
    doc_indices = []
    filter_stats = {'total': len(raw_docs), 'kept': 0, 'filtered_short': 0, 'filtered_empty': 0}

    for i, raw in enumerate(tqdm(raw_docs, desc="Preprocessing")):
        cleaned = preprocess_document(raw)
        if cleaned is None:
            filter_stats['filtered_short'] += 1
        elif len(cleaned.strip()) == 0:
            filter_stats['filtered_empty'] += 1
        else:
            cleaned_docs.append(cleaned)
            doc_indices.append(i)
            filter_stats['kept'] += 1

    logger.info(f"Preprocessing complete: {filter_stats['kept']}/{filter_stats['total']} docs kept "
                f"({filter_stats['filtered_short']} too short, {filter_stats['filtered_empty']} empty)")

    t_preprocess = time.time()

    # --- 3. Set up ChromaDB ---
    persist_dir = str(PROJECT_ROOT / config['chroma']['persist_directory'])
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = config['chroma']['collection_name']

    if args.force:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}' (--force)")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Check existing documents for idempotency
    existing_count = collection.count()
    if existing_count > 0 and not args.force:
        logger.info(f"Collection '{collection_name}' already has {existing_count} documents. "
                     "Use --force to re-index. Skipping ingestion.")
        return

    # --- 4. Embed ---
    logger.info(f"Loading embedding model: {config['model']['name']}...")
    model = SentenceTransformer(config['model']['name'])
    batch_size = config['model']['batch_size']

    logger.info(f"Encoding {len(cleaned_docs)} documents (batch_size={batch_size})...")
    embeddings = model.encode(
        cleaned_docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalise
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    t_embed = time.time()

    # --- 5. Upsert to ChromaDB ---
    logger.info("Upserting to ChromaDB...")
    upsert_batch = 500
    for start in tqdm(range(0, len(cleaned_docs), upsert_batch), desc="Upserting"):
        end = min(start + upsert_batch, len(cleaned_docs))
        batch_ids = []
        batch_docs = []
        batch_embeds = []
        batch_metas = []

        for j in range(start, end):
            orig_idx = doc_indices[j]
            doc_id = make_doc_id(cleaned_docs[j], orig_idx)
            batch_ids.append(doc_id)
            batch_docs.append(cleaned_docs[j])
            batch_embeds.append(embeddings[j].tolist())
            batch_metas.append({
                "doc_id": doc_id,
                "original_label": label_names[labels[orig_idx]],
                "label_index": int(labels[orig_idx]),
                "char_count": len(cleaned_docs[j]),
                "token_count": len(cleaned_docs[j].split()),
            })

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeds,
            metadatas=batch_metas,
        )

    t_upsert = time.time()

    # --- 6. Save embeddings for clustering ---
    clustering_dir = PROJECT_ROOT / config['clustering']['artefacts_dir']
    os.makedirs(clustering_dir, exist_ok=True)
    np.save(str(clustering_dir / 'embeddings.npy'), embeddings)

    # Save doc ID order for clustering → ChromaDB mapping
    doc_ids = [make_doc_id(cleaned_docs[j], doc_indices[j]) for j in range(len(cleaned_docs))]
    with open(str(clustering_dir / 'doc_id_order.json'), 'w') as f:
        json.dump(doc_ids, f)

    # --- 7. Ingestion report ---
    report = {
        "total_raw_documents": filter_stats['total'],
        "documents_kept": filter_stats['kept'],
        "documents_filtered_short": filter_stats['filtered_short'],
        "documents_filtered_empty": filter_stats['filtered_empty'],
        "embedding_dim": int(embeddings.shape[1]),
        "collection_name": collection_name,
        "final_collection_count": collection.count(),
        "timing": {
            "preprocessing_seconds": round(t_preprocess - t_start, 2),
            "embedding_seconds": round(t_embed - t_preprocess, 2),
            "upsert_seconds": round(t_upsert - t_embed, 2),
            "total_seconds": round(t_upsert - t_start, 2),
        }
    }

    report_path = PROJECT_ROOT / 'artefacts' / 'ingestion_report.json'
    with open(str(report_path), 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Ingestion complete! Report saved to {report_path}")
    logger.info(f"  Total time: {report['timing']['total_seconds']}s")
    logger.info(f"  Final collection count: {report['final_collection_count']}")


if __name__ == '__main__':
    main()
