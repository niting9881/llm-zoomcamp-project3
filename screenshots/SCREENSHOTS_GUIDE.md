# Dashboard Screenshots Guide

This guide helps you capture screenshots of the monitoring dashboards and UI for documentation.

## Required Screenshots

### 1. Prometheus Metrics Explorer
**URL**: http://localhost:9090/graph

**Recommended Queries to Capture**:

1. **Request Rate**
   ```promql
   rate(rag_app_requests_total[5m])
   ```
   Screenshot name: `prometheus_request_rate.png`

2. **Request Duration**
   ```promql
   rate(rag_app_request_duration_seconds_sum[5m]) / rate(rag_app_request_duration_seconds_count[5m])
   ```
   Screenshot name: `prometheus_request_duration.png`

3. **Active Requests**
   ```promql
   rag_app_active_requests
   ```
   Screenshot name: `prometheus_active_requests.png`

4. **Query Success vs Errors**
   ```promql
   rate(rag_app_queries_total[5m])
   rate(rag_app_query_errors_total[5m])
   ```
   Screenshot name: `prometheus_query_metrics.png`

### 2. Prometheus Targets
**URL**: http://localhost:9090/targets

**What to Capture**:
- Show both targets (rag-app and prometheus) in UP state
- Screenshot name: `prometheus_targets.png`

### 3. Streamlit UI
**URL**: http://localhost:8501

**Scenarios to Capture**:

1. **Query Interface**
   - Empty state with query input box
   - Screenshot name: `streamlit_ui_empty.png`

2. **Successful Query**
   - Example query: "What is pandas used for?"
   - Show the response with source attribution
   - Screenshot name: `streamlit_query_success.png`

3. **Different Package Query**
   - Example query: "How to use TensorFlow for neural networks?"
   - Screenshot name: `streamlit_tensorflow_query.png`

### 4. FastAPI Documentation
**URL**: http://localhost:8000/docs

**What to Capture**:
- Interactive API documentation showing all endpoints
- Expand the `/api/docs/search` endpoint
- Screenshot name: `fastapi_docs.png`

### 5. Grafana Dashboard (Optional)
**URL**: http://localhost:3000

**Setup Required**:
1. Login with `admin` / `admin`
2. Add Prometheus datasource
3. Create a dashboard with panels for:
   - Request rate
   - Request duration (p50, p95, p99)
   - Active requests gauge
   - Query success rate

**Screenshots**:
- `grafana_dashboard_overview.png`
- `grafana_request_metrics.png`

## Screenshot Requirements

### Technical Specifications
- **Format**: PNG or JPG
- **Resolution**: Minimum 1280x720 (720p)
- **Browser**: Use Chrome/Edge for consistent rendering
- **Window Size**: Maximize browser window for clarity

### Best Practices
1. **Clean UI**: Hide bookmarks bar and extra toolbars
2. **Zoom Level**: 100% browser zoom
3. **Timing**: Capture after running a few queries so graphs show data
4. **Annotations**: Consider adding arrows or highlights for key features
5. **Consistency**: Use same browser and theme for all screenshots

## How to Generate Test Data

Before capturing screenshots, generate some activity:

```bash
# Run a few test queries
curl "http://localhost:8000/api/docs/search?query=What+is+pandas"
curl "http://localhost:8000/api/docs/search?query=How+to+use+numpy"
curl "http://localhost:8000/api/docs/search?query=TensorFlow+tutorial"
curl "http://localhost:8000/api/docs/search?query=matplotlib+plotting"
curl "http://localhost:8000/api/docs/search?query=scikit-learn+classification"
```

Or use the Streamlit UI to ask several questions.

## Directory Structure

After capturing screenshots, your directory should look like:

```
screenshots/
├── SCREENSHOTS_GUIDE.md           # This file
├── prometheus_request_rate.png
├── prometheus_request_duration.png
├── prometheus_active_requests.png
├── prometheus_query_metrics.png
├── prometheus_targets.png
├── streamlit_ui_empty.png
├── streamlit_query_success.png
├── streamlit_tensorflow_query.png
├── fastapi_docs.png
└── grafana_dashboard_overview.png (optional)
```

## Adding Screenshots to README

Once captured, update the README.md to include screenshots:

```markdown
### Monitoring Dashboard Examples

#### Prometheus Metrics
![Prometheus Request Rate](screenshots/prometheus_request_rate.png)
*Request rate over time showing system usage patterns*

![Prometheus Targets](screenshots/prometheus_targets.png)
*All monitoring targets showing healthy status*

#### User Interface
![Streamlit Query Interface](screenshots/streamlit_query_success.png)
*Interactive query interface with successful response*

#### API Documentation
![FastAPI Interactive Docs](screenshots/fastapi_docs.png)
*Auto-generated API documentation*
```

## Screenshot Checklist

- [ ] Prometheus request rate graph
- [ ] Prometheus request duration graph
- [ ] Prometheus active requests graph
- [ ] Prometheus query metrics
- [ ] Prometheus targets status
- [ ] Streamlit UI (empty state)
- [ ] Streamlit successful query
- [ ] Streamlit different package query
- [ ] FastAPI documentation page
- [ ] Grafana dashboard (optional)

## Tips for High-Quality Screenshots

1. **Use Snipping Tool or Screenshot Apps**
   - Windows: Win + Shift + S
   - Mac: Cmd + Shift + 4
   - Chrome Extension: Awesome Screenshot

2. **Crop Appropriately**
   - Remove unnecessary browser chrome
   - Focus on the relevant content
   - Maintain aspect ratio

3. **Add Context**
   - Include timestamps if relevant
   - Show metric values clearly
   - Ensure text is readable

4. **Optimize File Size**
   - Use PNG for UI screenshots (sharp text)
   - Compress without losing quality
   - Keep under 500KB per image

## Automation Script (Optional)

Create a script to generate test load:

```python
# generate_test_load.py
import requests
import time

queries = [
    "What is pandas?",
    "How to use numpy arrays?",
    "TensorFlow neural networks tutorial",
    "matplotlib plotting examples",
    "scikit-learn classification",
    "beautifulsoup web scraping",
    "keras model training",
    "pytorch tensors"
]

for query in queries * 5:  # Repeat 5 times
    try:
        response = requests.get(
            "http://localhost:8000/api/docs/search",
            params={"query": query}
        )
        print(f"✓ Query: {query[:30]}... Status: {response.status_code}")
        time.sleep(2)
    except Exception as e:
        print(f"✗ Error: {e}")
```

Run with: `python generate_test_load.py`

---

**Last Updated**: 2025-10-18
