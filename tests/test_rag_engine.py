import pytest
from app.core.rag_engine import RAGEngine

def test_ingest_and_query():
    rag = RAGEngine()
    rag.ingest_docs()
    result = rag.query("What is BeautifulSoup?")
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0
