# 20 Newsgroups Semantic Search System

A production-grade semantic search system built on the 20 Newsgroups corpus (~18,800 documents, 20 categories). Features sentence embeddings, fuzzy clustering, cluster-aware semantic caching, and a FastAPI REST API.

## Architecture

```
POST /query → Embed → FCM Memberships → Cache Lookup → ChromaDB Retrieve → Cache Insert → Response
```

| Layer | Responsibility | Stack |
|-------|---------------|-------|
| **Embedding** | Encode documents & queries | `all-MiniLM-L6-v2`, ChromaDB |
| **Clustering** | Soft cluster assignments (FCM) | scikit-fuzzy, PCA |
| **Cache** | Paraphrase detection via cosine similarity | In-memory, cluster-bucketed |
| **API** | REST orchestration | FastAPI, Uvicorn |

> **Note on membership entropy:** Most documents show high membership entropy (mean=1.724, max possible=2.079), meaning the corpus is genuinely ambiguous — posts span multiple topics rather than belonging cleanly to one. This is a property of the data, not a model failure. The fuzzy clustering correctly reflects this uncertainty rather than forcing false precision.

## Quick Start

### Linux / macOS

```bash
# 1. Create and activate venv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest & embed corpus (~10 min, one-time)
python -m src.ingestion.ingest

# 4. Run FCM clustering (~5 min, one-time)
python -m src.clustering.fuzzy_cluster

# 5. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Windows (PowerShell)

```powershell
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest & embed corpus (~10 min, one-time)
python -m src.ingestion.ingest

# 4. Run FCM clustering (~5 min, one-time)
python -m src.clustering.fuzzy_cluster

# 5. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

> **Windows CMD** users: replace step 1 activation with `.venv\Scripts\activate.bat`

### Query Examples
```bash
# First query (cache miss)
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Are there any good telescope recommendations for beginners?"}'

# Paraphrase (cache hit)
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What telescope should a beginner astronomer buy?"}'

# Stats
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

## Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Semantic search with cache |
| `GET` | `/cache/stats` | Cache hit/miss statistics |
| `DELETE` | `/cache` | Flush cache |
| `GET` | `/health` | Health check |

## Configuration

All settings in `config.yaml`:
- **τ (similarity_threshold)**: 0.82 — the key cache tunable
- **bucket_cap**: 200 entries per cluster bucket
- **membership_threshold**: 0.15 — minimum FCM membership for bucket inclusion

## Testing

```bash
# Unit tests (no artefacts needed)
python -m pytest tests/test_preprocessing.py tests/test_semantic_cache.py -v

# Integration tests (requires artefacts)
python -m pytest tests/test_api.py -v
```

## Project Structure

```
├── src/
│   ├── ingestion/       # Load → preprocess → embed → ChromaDB
│   ├── clustering/      # PCA + FCM training & analysis
│   ├── cache/           # Semantic cache & threshold analysis
│   └── api/             # FastAPI app, routes, state, schemas
├── artefacts/           # Persisted models & data
├── tests/               # Unit & integration tests
├── config.yaml          # System configuration
├── Dockerfile
└── docker-compose.yml
```
