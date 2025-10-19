
import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI
from app.api.routes import router
import os
import time

dotenv.load_dotenv()
import logging
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry, make_asgi_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_app")

# Prometheus metrics
REQUEST_COUNT = Counter(
    'rag_app_requests_total', 
    'Total number of requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'rag_app_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)
ACTIVE_REQUESTS = Gauge(
    'rag_app_active_requests',
    'Number of active requests'
)
QUERY_COUNT = Counter(
    'rag_app_queries_total',
    'Total number of RAG queries processed'
)
QUERY_ERRORS = Counter(
    'rag_app_query_errors_total',
    'Total number of query errors'
)

app = FastAPI(title="Python Documentation RAG Helper")

# Middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    finally:
        ACTIVE_REQUESTS.dec()

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "Internal server error. Please contact support."}
    )

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
