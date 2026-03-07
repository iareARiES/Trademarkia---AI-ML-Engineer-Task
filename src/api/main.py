"""
FastAPI application entry point with lifespan context manager.
Usage: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.api.state import initialise_state

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — loads all models and artefacts on startup.
    If any component fails to load, the app raises immediately (no degraded state).
    """
    logger.info("=" * 60)
    logger.info("Starting 20 Newsgroups Semantic Search API...")
    logger.info("=" * 60)

    try:
        initialise_state()
        logger.info("All components loaded. API is ready.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialise application state: {e}")
        raise RuntimeError(f"Cannot start API — initialisation failed: {e}") from e

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="20 Newsgroups Semantic Search",
    description="Semantic search with fuzzy clustering and intelligent caching",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "semantic-search"}
