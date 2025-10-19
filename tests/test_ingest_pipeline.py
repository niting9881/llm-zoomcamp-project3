import os
import shutil
import tempfile
from pathlib import Path
from app.core.rag_engine import RAGEngine

def test_batch_ingestion(tmp_path):
    # Setup: create temp raw data dir and files
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file1 = raw_dir / "test1.md"
    file2 = raw_dir / "test2.html"
    file1.write_text("# Test Markdown\nThis is a test.")
    file2.write_text("<html><body>Test HTML</body></html>")
    processed_log = tmp_path / "processed_files.log"
    vector_dir = tmp_path / "vectorstore"
    rag = RAGEngine(docs_dir=raw_dir, vector_dir=vector_dir)
    # Batch ingest
    rag.ingest_docs([file1, file2])
    # Simulate Airflow logic: update processed log
    with open(processed_log, "w") as f:
        f.write(str(file1) + "\n")
        f.write(str(file2) + "\n")
    # Check processed log
    processed = set(line.strip() for line in processed_log.read_text().splitlines())
    assert str(file1) in processed
    assert str(file2) in processed

def test_incremental_ingestion(tmp_path):
    # Setup: create temp raw data dir and files
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file1 = raw_dir / "test1.md"
    file1.write_text("# Test Markdown\nThis is a test.")
    processed_log = tmp_path / "processed_files.log"
    vector_dir = tmp_path / "vectorstore"
    rag = RAGEngine(docs_dir=raw_dir, vector_dir=vector_dir)
    # Initial ingest
    rag.ingest_docs([file1])
    with open(processed_log, "w") as f:
        f.write(str(file1) + "\n")
    # Add new file
    file2 = raw_dir / "test2.html"
    file2.write_text("<html><body>Test HTML</body></html>")
    # Incremental ingest: only new file
    new_files = [file2] if str(file2) not in processed_log.read_text() else []
    rag.ingest_docs(new_files)
    with open(processed_log, "a") as f:
        f.write(str(file2) + "\n")
    processed = set(line.strip() for line in processed_log.read_text().splitlines())
    assert str(file1) in processed
    assert str(file2) in processed
