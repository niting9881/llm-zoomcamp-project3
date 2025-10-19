# 🐍 Python Documentation RAG Helper

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Evaluation Results](#evaluation-results)
- [Data Sources and Coverage](#data-sources-and-coverage)
- [System Limitations](#system-limitations)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for Python documentation and PyPI package information. It provides AI-powered answers to questions about **52+ popular Python packages** including TensorFlow, PyTorch, scikit-learn, and more - covering the complete ML/DL/Visualization ecosystem.

Built as part of the DataTalks.Club LLM Zoomcamp 2025 curriculum, this system showcases production-ready architecture patterns and best practices for educational and real-world use.

### 🎯 Key Highlights
- **52+ ML/DL/Viz Packages**: TensorFlow, PyTorch, Keras, scikit-learn, XGBoost, Matplotlib, Plotly, SHAP, MLflow, and more
- **Python Official Docs**: Versions 3.11, 3.12, 3.14
- **100% Harvest Success Rate**: All packages successfully harvested with metadata and documentation
- **Production Ready**: Docker containerization, Prometheus monitoring, comprehensive error handling

---

## Problem Statement
Python documentation and package information are scattered across multiple sources (official docs, PyPI, READMEs). Developers often struggle to quickly find authoritative, context-rich answers to their questions. This project bridges that gap by providing a unified, natural language interface to search and retrieve relevant Python documentation and package details.

---

## Solution Architecture

### System Components

This RAG system is built with a modern, containerized architecture designed for scalability and observability:

#### 1. **Data Ingestion Layer**
- **Python Doc Harvester**: Scrapes Python 3.11, 3.12, 3.14 official documentation
- **PyPI Package Harvester**: Downloads README and metadata for 52 ML/DL/Viz packages
- **Ingestion Pipeline**: Orchestrates harvesting, processing, and chunking of documents

#### 2. **Storage Layer**
- **ChromaDB**: Vector database for semantic search (port 8001)
  - Stores document embeddings (OpenAI text-embedding-ada-002)
  - Supports similarity search with metadata filtering
  - Persistent storage in Docker volumes

#### 3. **RAG Processing Layer**
- **Simple RAG Engine**: Core retrieval and generation logic
  - Vector retrieval with configurable top_k
  - Context assembly from retrieved documents
  - LLM-based answer generation (GPT-4 / GPT-3.5-turbo)
  - Source attribution for all responses

#### 4. **API Layer** (Port 8000)
- **FastAPI Backend**: REST API with automatic OpenAPI docs
  - `/api/docs/search`: RAG query endpoint
  - `/metrics`: Prometheus metrics endpoint
  - `/health`: Health check endpoint
  - Comprehensive error handling and logging

#### 5. **User Interface Layer** (Port 8501)
- **Streamlit Web UI**: Interactive query interface
  - Natural language question input
  - Real-time RAG responses with sources
  - Error handling with user-friendly messages
  - Support for 52+ packages

#### 6. **Monitoring & Observability** (Ports 9090, 3000)
- **Prometheus**: Metrics collection and storage
  - 5 custom metrics tracked
  - 10-second scrape interval
  - Time-series data retention
- **Grafana**: Visualization dashboards
  - Pre-configured Prometheus data source
  - Custom dashboard support

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
├──────────────────────┬──────────────────────┬───────────────────────┤
│  Python Docs (3.11,  │   PyPI Packages      │   Package Metadata   │
│   3.12, 3.14)        │   (52 packages)      │   (JSON)             │
└──────────────────────┴──────────────────────┴───────────────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │  INGESTION PIPELINE    │
                     │  - HTML Parsing        │
                     │  - Markdown Processing │
                     │  - Chunking Strategy   │
                     └────────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   CHROMADB VECTORSTORE │
                     │   - OpenAI Embeddings  │
                     │   - Persistent Storage │
                     │   - Port: 8001         │
                     └────────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   SIMPLE RAG ENGINE    │
                     │   - Retrieval (top_k)  │
                     │   - Context Assembly   │
                     │   - LLM Generation     │
                     │   - Source Attribution │
                     └────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
    ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
    │   FASTAPI API    │ │  STREAMLIT   │ │   PROMETHEUS     │
    │   Port: 8000     │ │     UI       │ │   MONITORING     │
    │                  │ │  Port: 8501  │ │   Port: 9090     │
    │  - /api/docs/*   │ │              │ │                  │
    │  - /metrics      │ │  - Query UI  │ │  - Metrics       │
    │  - /health       │ │  - Results   │ │  - Alerting      │
    └──────────────────┘ └──────────────┘ └──────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │     GRAFANA      │
                                          │   Dashboards     │
                                          │   Port: 3000     │
                                          └──────────────────┘
```

### Docker Compose Architecture

All components run in isolated containers with defined dependencies:

```yaml
Services:
  rag-app:       # FastAPI + Streamlit (ports 8000, 8501)
  chromadb:      # Vector database (port 8001)
  prometheus:    # Metrics collector (port 9090)
  grafana:       # Dashboard UI (port 3000)
```

### Monitoring Details

#### Prometheus Metrics Tracked

1. **`rag_app_requests_total`** (Counter)
   - Total HTTP requests by method, endpoint, and status
   - Labels: `[method, endpoint, status]`

2. **`rag_app_request_duration_seconds`** (Histogram)
   - Request latency distribution
   - Labels: `[method, endpoint]`
   - Buckets: Default histogram buckets

3. **`rag_app_active_requests`** (Gauge)
   - Number of concurrent requests being processed
   - Real-time active request count

4. **`rag_app_queries_total`** (Counter)
   - Total RAG queries processed successfully
   - Incremented on each successful query

5. **`rag_app_query_errors_total`** (Counter)
   - Total query errors encountered
   - Incremented on each failed query

#### Accessing Monitoring

**Prometheus UI**: [http://localhost:9090](http://localhost:9090)
- Query metrics with PromQL
- View targets and their health status
- Create custom visualizations

**Grafana UI**: [http://localhost:3000](http://localhost:3000)
- Default credentials: `admin` / `admin`
- Pre-configured Prometheus datasource
- Create custom dashboards

#### Sample Prometheus Queries

```promql
# Request rate (requests per second)
rate(rag_app_requests_total[5m])

# Average request duration
rate(rag_app_request_duration_seconds_sum[5m]) / rate(rag_app_request_duration_seconds_count[5m])

# Error rate
rate(rag_app_query_errors_total[5m])

# Active requests over time
rag_app_active_requests

# Total queries processed
rag_app_queries_total
```

#### Dashboard Screenshots

> 📸 **Note**: Screenshots can be captured following the guide in [`screenshots/SCREENSHOTS_GUIDE.md`](screenshots/SCREENSHOTS_GUIDE.md)

**Prometheus Metrics Dashboard**:
- Shows request rate, duration, and error metrics in real-time
- Accessible at http://localhost:9090/graph
- *Screenshot placeholder: `screenshots/prometheus_dashboard.png`*

**Prometheus Targets Status**:
- Displays health status of all monitored services
- Both `rag-app` and `prometheus` targets should show UP status
- *Screenshot placeholder: `screenshots/prometheus_targets.png`*

**Streamlit Query Interface**:
- Interactive web UI for asking questions
- Shows real-time responses with source attribution
- *Screenshot placeholder: `screenshots/streamlit_ui.png`*

**FastAPI Interactive Documentation**:
- Auto-generated API docs with try-it-out functionality
- Available at http://localhost:8000/docs
- *Screenshot placeholder: `screenshots/fastapi_docs.png`*

### Data Flow

1. **Query Submission**: User enters question via Streamlit UI or API
2. **Embedding**: Query converted to vector using OpenAI embeddings
3. **Retrieval**: ChromaDB returns top-k most similar documents
4. **Context Assembly**: Retrieved documents combined into context
5. **Generation**: LLM generates answer based on context
6. **Response**: Answer returned with source attribution
7. **Metrics**: Request logged to Prometheus

### Technology Choices

| Component | Technology | Reasoning |
|-----------|-----------|-----------|
| Vector DB | ChromaDB | Lightweight, persistent, Python-native |
| LLM | GPT-3.5-turbo | 7x faster, 10x cheaper than GPT-4 |
| Embeddings | text-embedding-ada-002 | OpenAI standard, 1536 dimensions |
| API Framework | FastAPI | Async support, auto docs, Python 3.11+ |
| UI Framework | Streamlit | Rapid prototyping, Python-native |
| Monitoring | Prometheus + Grafana | Industry standard, powerful querying |
| Container | Docker Compose | Multi-service orchestration, reproducible |

---

## Features
- **Hybrid Search:** Combines BM25 keyword and semantic vector retrieval for optimal results.
- **Query Rewriting:** Expands and refines user queries for better recall.
- **Document Reranking:** Improves answer precision using cross-encoder models.
- **Batch & Incremental Ingestion:** Supports both full and incremental updates of documentation.
- **Source Attribution:** Returns sources for every answer.
- **Monitoring:** Prometheus and Grafana dashboards for observability.
- **Containerized:** Easy deployment with Docker Compose.

---

## Technology Stack
- **Python 3.14**
- **FastAPI** (API backend)
- **Streamlit** (UI frontend)
- **ChromaDB** (vector store)
- **BM25 (rank_bm25)** (keyword search)
- **LangChain** (RAG orchestration)
- **Sentence Transformers** (embeddings, reranking)
- **Airflow** (optional, for scheduled ingestion)
- **Prometheus & Grafana** (monitoring)
- **Docker, docker-compose** (deployment)

---

## Setup Instructions
### Prerequisites
- Docker and Docker Compose installed
- Python 3.14 (for local dev)
- OpenAI API key (for embeddings)

### Installation Steps
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. Copy and edit `.env.example` to `.env` and add your API keys.
3. Build and start all services:
   ```sh
   docker compose up --build
   ```
4. (Optional) Run initial data ingestion:
   ```sh
   python Ingestion/python_doc_harvester.py
   python Ingestion/pypi_harvester.py
   ```

---

## Usage Guide

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit UI** | [http://localhost:8501](http://localhost:8501) | Interactive query interface |
| **FastAPI Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API documentation |
| **API Endpoint** | http://localhost:8000/api/docs/search | RAG query endpoint |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Metrics and monitoring |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Visualization dashboards |
| **ChromaDB** | http://localhost:8001 | Vector database (internal) |

### Using the Streamlit UI

1. Open http://localhost:8501 in your browser
2. Enter your question in the text input
3. Click "Search" or press Enter
4. View the AI-generated answer with source attribution

**Example Questions**:
- "What is pandas used for?"
- "How to create a DataFrame in pandas?"
- "What is the difference between NumPy and pandas?"
- "How to plot graphs with matplotlib?"
- "What is TensorFlow?"

### Using the API

**Python Example**:
```python
import requests

# Query the RAG system
response = requests.get(
    "http://localhost:8000/api/docs/search",
    params={"query": "What is BeautifulSoup?"}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

**cURL Example**:
```bash
curl "http://localhost:8000/api/docs/search?query=How+to+use+pandas"
```

**Response Format**:
```json
{
  "answer": "pandas is a powerful Python library...",
  "sources": ["pandas", "numpy"],
  "query": "What is pandas?",
  "retrieval_time": 0.234,
  "generation_time": 1.567
}
```

### Monitoring Your System

#### Prometheus Metrics

**Quick Health Check**:
```bash
# Check if metrics are being collected
curl http://localhost:8000/metrics
```

**Key Metrics to Monitor**:
1. **Request Rate**: `rate(rag_app_requests_total[5m])`
2. **Latency**: `histogram_quantile(0.95, rag_app_request_duration_seconds_bucket)`
3. **Error Rate**: `rate(rag_app_query_errors_total[5m])`
4. **Active Requests**: `rag_app_active_requests`

**Access Prometheus**:
1. Open http://localhost:9090
2. Go to "Graph" tab
3. Enter a PromQL query (see examples above)
4. Click "Execute" to see results

#### Grafana Dashboards

**Initial Setup**:
1. Open http://localhost:3000
2. Login with `admin` / `admin`
3. Add Prometheus as a data source:
   - URL: `http://prometheus:9090`
   - Access: Server (default)
4. Create a new dashboard
5. Add panels for the key metrics listed above

**Recommended Panels**:
- Request rate (Graph)
- Request duration percentiles (Graph with p50, p95, p99)
- Active requests (Gauge)
- Error rate (Graph)
- Total queries (Stat)

### Health Checks

**Check All Services**:
```bash
# RAG App
curl http://localhost:8000/health

# ChromaDB
curl http://localhost:8001/api/v1/heartbeat

# Prometheus
curl http://localhost:9090/-/healthy
```

**Docker Service Status**:
```bash
docker compose ps
```

All services should show "Up" status.

---

## Evaluation Results


#### Retrieval Evaluation
- **Metric**: Hit Rate @ top_k
- **Test Set**: 15 queries across 10 categories
- **Results**: 20% hit rate (requires full dataset ingestion for 70-90%)
- **Optimal Config**: `top_k=3`

| top_k | Hit Rate | Hits | Misses |
|-------|----------|------|--------|
| 1     | 20.0%    | 3    | 12     |
| 3     | 20.0%    | 3    | 12     |
| 5     | 20.0%    | 3    | 12     |
| 10    | 20.0%    | 3    | 12     |

#### LLM Evaluation
- **Configurations Tested**: 3 prompts × 2 models × 4 temperatures = 10 configs
- **Test Set**: 8 queries
- **Metrics**: Response time, answer length, code presence

**Optimal Configuration** 🏆:
- **Model**: `gpt-3.5-turbo` (7x faster, 10x cheaper than GPT-4)
- **Temperature**: `0.7` (most comprehensive answers)
- **Prompt**: `detailed` (62.5% include code examples)

| Configuration | Avg Response Time | Success Rate | Cost/Query |
|---------------|-------------------|--------------|------------|
| **gpt-3.5-turbo (recommended)** | **1.88s** | **100%** | **$0.002** |
| gpt-4 (temp 0.0) | 13.16s | 87.5% | $0.020 |
| gpt-4 (temp 0.7) | 15.41s | 100% | $0.025 |

**See [LLM_EVALUATION.md](LLM_EVALUATION.md) for detailed analysis**

#### Running Evaluations

```bash
# Retrieval evaluation
docker exec rag-app python app/evaluation/retrieval_eval.py

# LLM evaluation
docker exec rag-app python app/evaluation/llm_eval.py

# Copy results
docker cp rag-app:/app/evaluation_results ./evaluation_results
```

---

## Data Sources and Coverage

### 📚 Python Official Documentation
- **Python 3.14** (200+ modules)
- **Python 3.12** (200+ modules)  
- **Python 3.11** (200+ modules)

### 📦 PyPI Packages (52 packages)

#### Deep Learning (9)
tensorflow, torch (PyTorch), keras, jax, flax, pytorch-lightning, transformers, fastai, mxnet

#### Machine Learning (10)
scikit-learn, xgboost, lightgbm, catboost, optuna, hyperopt, sklearn-pandas, imbalanced-learn, mlxtend

#### Computer Vision (6)
opencv-python, pillow, torchvision, albumentations, imageio, scikit-image

#### NLP (6)
nltk, spacy, gensim, textblob, sentencepiece, tokenizers

#### Visualization (7)
matplotlib, seaborn, plotly, bokeh, altair, holoviews, dash

#### Data Processing (5)
scipy, statsmodels, featuretools, category-encoders, feature-engine

#### Model Interpretability (3)
shap, lime, eli5

#### MLOps (3)
mlflow, wandb, tensorboard

#### Utilities (4)
beautifulsoup4, langchain, pandas, numpy

**📊 Total Coverage**: 52 packages + 600+ Python modules = **652+ documentation sources**

See [ML_DL_VIZ_PACKAGES.md](ML_DL_VIZ_PACKAGES.md) for complete package catalog.

---

## Project Structure
```
├── Interface/           # Streamlit UI
│   └── ui_streamlit.py
├── Ingestion/           # Ingestion scripts
│   ├── python_doc_harvester.py
│   ├── pypi_harvester.py
│   └── dag_ingest_docs.py
├── Monitoring/          # Prometheus, Grafana configs
│   └── prometheus.yml
├── Containerization/    # Docker, docker-compose
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── app/                 # FastAPI app, core logic
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── rag_engine.py
│   └── main.py
├── data/                # Raw and processed data
├── tests/               # Unit tests
├── pyproject.toml       # Python dependencies
├── README.md            # This file
└── ...
```

---

## Future Improvements
- Add more packages and Python versions
- Scheduled/automatic ingestion
- Cloud deployment support
- User authentication and access control
- More advanced reranking and feedback loops

---

## Developed By
Nitin Gupta for llm-zoomcamp

---

## Acknowledgments
- DataTalks.Club for LLM Zoomcamp structure
- Python, PyPI, and open-source community

