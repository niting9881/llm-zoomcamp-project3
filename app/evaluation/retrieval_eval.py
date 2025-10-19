"""
Retrieval Evaluation Script

Evaluates different retrieval approaches to find the best configuration.
Tests multiple top_k values and measures hit rate.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.rag_engine_simple import SimpleRAGEngine
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Ground truth test set with known answers
TEST_QUERIES = [
    {
        "query": "What is BeautifulSoup used for?",
        "expected_package": "beautifulsoup4",
        "category": "Web Scraping"
    },
    {
        "query": "How to create a pandas DataFrame?",
        "expected_package": "pandas",
        "category": "Data Processing"
    },
    {
        "query": "What is NumPy?",
        "expected_package": "numpy",
        "category": "Numerical Computing"
    },
    {
        "query": "How to use TensorFlow for deep learning?",
        "expected_package": "tensorflow",
        "category": "Deep Learning"
    },
    {
        "query": "What is PyTorch?",
        "expected_package": "torch",
        "category": "Deep Learning"
    },
    {
        "query": "How to plot graphs with matplotlib?",
        "expected_package": "matplotlib",
        "category": "Visualization"
    },
    {
        "query": "What is scikit-learn?",
        "expected_package": "scikit-learn",
        "category": "Machine Learning"
    },
    {
        "query": "How to use XGBoost for classification?",
        "expected_package": "xgboost",
        "category": "Machine Learning"
    },
    {
        "query": "What is SHAP and how does it explain models?",
        "expected_package": "shap",
        "category": "Interpretability"
    },
    {
        "query": "How to track ML experiments with MLflow?",
        "expected_package": "mlflow",
        "category": "MLOps"
    },
    {
        "query": "How to create interactive visualizations with Plotly?",
        "expected_package": "plotly",
        "category": "Visualization"
    },
    {
        "query": "What is spaCy used for?",
        "expected_package": "spacy",
        "category": "NLP"
    },
    {
        "query": "How to use OpenCV for image processing?",
        "expected_package": "opencv-python",
        "category": "Computer Vision"
    },
    {
        "query": "What is LightGBM?",
        "expected_package": "lightgbm",
        "category": "Machine Learning"
    },
    {
        "query": "How to optimize hyperparameters with Optuna?",
        "expected_package": "optuna",
        "category": "Machine Learning"
    },
]


def evaluate_retrieval(engine, top_k=5):
    """
    Evaluate retrieval quality using hit rate metric.
    
    Args:
        engine: RAG engine instance
        top_k: Number of top documents to retrieve
        
    Returns:
        tuple: (hit_rate, results_list)
    """
    hits = 0
    results = []
    
    logger.info(f"Evaluating retrieval with top_k={top_k}")
    
    # Load vector store if needed
    if engine.vector_store is None:
        if engine.vector_dir.exists():
            logger.info("Loading existing vector store...")
            from langchain_community.vectorstores import Chroma
            engine.vector_store = Chroma(
                persist_directory=str(engine.vector_dir),
                embedding_function=engine.embeddings,
                collection_name="python_docs"
            )
        else:
            raise ValueError("No vector store found. Run ingestion first.")
    
    # Create retriever
    retriever = engine.vector_store.as_retriever(search_kwargs={"k": top_k})
    
    for i, test in enumerate(TEST_QUERIES, 1):
        try:
            # Retrieve documents
            docs = retriever.invoke(test["query"])
            
            # Check if expected package is in retrieved documents
            hit = False
            for doc in docs:
                source = doc.metadata.get('source', '')
                if test["expected_package"].lower() in source.lower():
                    hit = True
                    break
            
            if hit:
                hits += 1
            
            results.append({
                "query": test["query"],
                "expected": test["expected_package"],
                "category": test["category"],
                "hit": hit,
                "top_k": top_k
            })
            
            logger.debug(f"[{i}/{len(TEST_QUERIES)}] Query: {test['query'][:50]}... Hit: {hit}")
            
        except Exception as e:
            logger.error(f"Error evaluating query '{test['query']}': {e}")
            results.append({
                "query": test["query"],
                "expected": test["expected_package"],
                "category": test["category"],
                "hit": False,
                "top_k": top_k,
                "error": str(e)
            })
    
    hit_rate = hits / len(TEST_QUERIES) if TEST_QUERIES else 0
    logger.info(f"Hit Rate @{top_k}: {hit_rate:.2%} ({hits}/{len(TEST_QUERIES)})")
    
    return hit_rate, results


def compare_top_k_values():
    """
    Compare different top_k values to find optimal configuration.
    
    Returns:
        DataFrame with comparison results
    """
    print("=" * 80)
    print("RETRIEVAL EVALUATION - Comparing top_k Values")
    print("=" * 80)
    print(f"\nTest set size: {len(TEST_QUERIES)} queries")
    print(f"Categories: {len(set(t['category'] for t in TEST_QUERIES))} categories\n")
    
    # Initialize engine once
    logger.info("Initializing RAG engine...")
    engine = SimpleRAGEngine()
    
    # Test different top_k values
    top_k_values = [1, 3, 5, 10]
    comparison_results = []
    all_detailed_results = []
    
    for top_k in top_k_values:
        hit_rate, detailed_results = evaluate_retrieval(engine, top_k)
        
        comparison_results.append({
            "top_k": top_k,
            "hit_rate": hit_rate,
            "hits": sum(1 for r in detailed_results if r['hit']),
            "misses": sum(1 for r in detailed_results if not r['hit'])
        })
        
        all_detailed_results.extend(detailed_results)
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_results)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    
    # Find best configuration
    best_config = df_comparison.loc[df_comparison['hit_rate'].idxmax()]
    print(f"\n✅ Best Configuration: top_k={int(best_config['top_k'])} with {best_config['hit_rate']:.2%} hit rate")
    
    # Category breakdown for best config
    df_detailed = pd.DataFrame(all_detailed_results)
    best_k = int(best_config['top_k'])
    best_k_results = df_detailed[df_detailed['top_k'] == best_k]
    
    category_performance = best_k_results.groupby('category')['hit'].agg(['sum', 'count', 'mean'])
    category_performance.columns = ['hits', 'total', 'hit_rate']
    category_performance = category_performance.sort_values('hit_rate', ascending=False)
    
    print(f"\n" + "=" * 80)
    print(f"PERFORMANCE BY CATEGORY (top_k={best_k})")
    print("=" * 80)
    print(category_performance)
    
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    df_comparison.to_csv(output_dir / "retrieval_comparison.csv", index=False)
    df_detailed.to_csv(output_dir / "retrieval_detailed.csv", index=False)
    
    print(f"\n📊 Results saved to:")
    print(f"  - evaluation_results/retrieval_comparison.csv")
    print(f"  - evaluation_results/retrieval_detailed.csv")
    
    return df_comparison, df_detailed


def analyze_misses(df_detailed, top_k=5):
    """
    Analyze queries that failed to retrieve expected documents.
    
    Args:
        df_detailed: DataFrame with detailed results
        top_k: The top_k value to analyze
    """
    misses = df_detailed[(df_detailed['top_k'] == top_k) & (~df_detailed['hit'])]
    
    if len(misses) == 0:
        print(f"\n🎉 Perfect score at top_k={top_k}! No misses to analyze.")
        return
    
    print(f"\n" + "=" * 80)
    print(f"MISSED QUERIES ANALYSIS (top_k={top_k})")
    print("=" * 80)
    print(f"Total misses: {len(misses)}\n")
    
    for _, miss in misses.iterrows():
        print(f"❌ Query: {miss['query']}")
        print(f"   Expected: {miss['expected']}")
        print(f"   Category: {miss['category']}")
        print()


def main():
    """Main evaluation function."""
    try:
        # Run comparison
        df_comparison, df_detailed = compare_top_k_values()
        
        # Analyze misses for best configuration
        best_k = int(df_comparison.loc[df_comparison['hit_rate'].idxmax(), 'top_k'])
        analyze_misses(df_detailed, best_k)
        
        print("\n" + "=" * 80)
        print("✅ RETRIEVAL EVALUATION COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review evaluation_results/retrieval_comparison.csv")
        print("2. Check evaluation_results/retrieval_detailed.csv for per-query analysis")
        print("3. Update RAG engine to use optimal top_k value")
        print("4. Consider implementing hybrid search for further improvement")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
