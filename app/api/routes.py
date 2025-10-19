from fastapi import APIRouter, Query
from typing import Optional
from app.core.rag_engine_simple import SimpleRAGEngine
import logging
from prometheus_client import Counter

router = APIRouter()
rag_engine = SimpleRAGEngine()
logger = logging.getLogger("rag_api")

# Metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total RAG queries')
ERROR_COUNTER = Counter('rag_errors_total', 'Total RAG errors', ['error_type'])

@router.get("/docs/search")
def search_docs(query: str = Query(..., description="Search query for documentation"), package: Optional[str] = Query(None, description="Optional package name")):
    try:
        QUERY_COUNTER.inc()
        
        # Optionally filter docs by package (not implemented in this quickstart)
        # Ingest docs if vector store is not initialized
        if rag_engine.vector_store is None:
            logger.info("Ingesting documentation files...")
            rag_engine.ingest_docs()
        logger.info(f"Querying RAG engine with: {query}")
        result = rag_engine.query(query)
        return {
            "query": query,
            "package": package,
            "result": result
        }
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        logger.error(f"Error in RAG engine: {e}", exc_info=True)
        return {
            "error": str(e),
            "query": query,
            "package": package
        }
