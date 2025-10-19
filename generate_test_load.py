"""
Generate test load for monitoring dashboards

This script sends multiple queries to the RAG system to generate
metrics for Prometheus and Grafana dashboards.

Usage:
    python generate_test_load.py
"""

import requests
import time
import random
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/api/docs/search"

# Test queries
QUERIES = [
    "What is pandas?",
    "How to use numpy arrays?",
    "TensorFlow neural networks tutorial",
    "matplotlib plotting examples",
    "scikit-learn classification",
    "beautifulsoup web scraping",
    "keras model training",
    "pytorch tensors",
    "What is the difference between NumPy and pandas?",
    "How to create a DataFrame?",
    "Deep learning with TensorFlow",
    "Data visualization with seaborn",
    "Machine learning with scikit-learn",
    "Web scraping with requests",
    "Natural language processing with spaCy",
    "Computer vision with OpenCV",
    "Model interpretability with SHAP",
    "MLOps with MLflow",
    "Hyperparameter tuning with Optuna",
    "Data preprocessing with pandas"
]

def send_query(query: str) -> dict:
    """Send a single query to the API"""
    try:
        start_time = time.time()
        response = requests.get(
            API_URL,
            params={"query": query},
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            return {
                "status": "success",
                "query": query,
                "elapsed": elapsed,
                "status_code": response.status_code
            }
        else:
            return {
                "status": "error",
                "query": query,
                "elapsed": elapsed,
                "status_code": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "elapsed": None,
            "error": str(e)
        }

def generate_load(num_queries: int = 50, delay_range: tuple = (1, 3)):
    """
    Generate test load by sending multiple queries
    
    Args:
        num_queries: Number of queries to send
        delay_range: (min, max) delay between queries in seconds
    """
    print(f"🚀 Starting test load generation")
    print(f"📊 Target: {num_queries} queries")
    print(f"⏱️  Delay: {delay_range[0]}-{delay_range[1]} seconds between queries")
    print(f"🕐 Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 70)
    
    results = {
        "success": 0,
        "errors": 0,
        "total_time": 0
    }
    
    for i in range(num_queries):
        # Select random query
        query = random.choice(QUERIES)
        
        # Send query
        result = send_query(query)
        
        # Update results
        if result["status"] == "success":
            results["success"] += 1
            icon = "✓"
            color = "\033[92m"  # Green
        else:
            results["errors"] += 1
            icon = "✗"
            color = "\033[91m"  # Red
        
        if result["elapsed"]:
            results["total_time"] += result["elapsed"]
        
        # Print result
        reset = "\033[0m"
        status = result.get("status_code", "ERROR")
        elapsed_str = f"{result['elapsed']:.2f}s" if result["elapsed"] else "N/A"
        print(f"{color}{icon}{reset} [{i+1}/{num_queries}] {query[:40]:<40} | {status} | {elapsed_str}")
        
        # Random delay before next query
        if i < num_queries - 1:
            delay = random.uniform(*delay_range)
            time.sleep(delay)
    
    # Print summary
    print("-" * 70)
    print(f"📈 SUMMARY")
    print(f"✓ Success: {results['success']}/{num_queries} ({results['success']/num_queries*100:.1f}%)")
    print(f"✗ Errors: {results['errors']}/{num_queries} ({results['errors']/num_queries*100:.1f}%)")
    if results['success'] > 0:
        avg_time = results['total_time'] / results['success']
        print(f"⏱️  Average response time: {avg_time:.2f}s")
    print(f"🕐 End time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"⏳ Total duration: {results['total_time']:.1f}s")
    print("-" * 70)
    print(f"✅ Test load generation complete!")
    print(f"📊 Check Prometheus: http://localhost:9090/graph")
    print(f"📈 Check Grafana: http://localhost:3000")

def burst_load(num_concurrent: int = 5):
    """
    Generate burst load (multiple concurrent requests)
    
    Args:
        num_concurrent: Number of concurrent requests to send
    """
    import concurrent.futures
    
    print(f"💥 Generating burst load with {num_concurrent} concurrent requests")
    print("-" * 70)
    
    queries = random.sample(QUERIES, min(num_concurrent, len(QUERIES)))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(send_query, query) for query in queries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"✓ Success: {success}/{num_concurrent}")
    print(f"✗ Errors: {errors}/{num_concurrent}")
    print("-" * 70)

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("🧪 RAG System Test Load Generator")
    print("=" * 70)
    print()
    
    # Check if API is accessible
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✓ API is accessible")
    except Exception as e:
        print(f"✗ API is not accessible: {e}")
        print("⚠️  Make sure the RAG system is running: docker compose up")
        sys.exit(1)
    
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "burst":
            burst_load(num_concurrent=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
        else:
            num_queries = int(sys.argv[1])
            generate_load(num_queries=num_queries)
    else:
        # Default: 50 queries
        generate_load(num_queries=50)
    
    print()
    print("💡 Tip: Run 'python generate_test_load.py 100' for 100 queries")
    print("💡 Tip: Run 'python generate_test_load.py burst 10' for 10 concurrent requests")
