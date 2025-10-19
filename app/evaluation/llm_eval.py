"""
LLM Evaluation Script

Evaluates different LLM configurations to find the best approach.
Tests different prompts, models, and temperatures.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.rag_engine_simple import SimpleRAGEngine
from langchain_openai import ChatOpenAI
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Test queries for LLM evaluation
TEST_QUERIES = [
    "What is BeautifulSoup used for?",
    "How to create a pandas DataFrame?",
    "What is NumPy and why use it?",
    "How to use TensorFlow for neural networks?",
    "What is the difference between scikit-learn and XGBoost?",
    "How to create visualizations with matplotlib?",
    "What is SHAP?",
    "How to track experiments with MLflow?",
]


# Different prompt templates to test
PROMPT_TEMPLATES = {
    "simple": """Context: {context}

Question: {question}

Answer:""",
    
    "detailed": """You are a helpful Python documentation assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context. Include code examples if relevant.""",
    
    "structured": """You are a Python expert. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer based only on the context provided
- Be concise but complete
- Include code examples when relevant
- If the context doesn't contain the answer, say so

ANSWER:""",
}


def evaluate_prompt_variant(engine, prompt_template, query):
    """
    Evaluate a single prompt variant on a query.
    
    Args:
        engine: RAG engine instance
        prompt_template: Prompt template string
        query: Test query
        
    Returns:
        dict with evaluation metrics
    """
    try:
        # Load vector store if needed
        if engine.vector_store is None:
            if engine.vector_dir.exists():
                from langchain_community.vectorstores import Chroma
                engine.vector_store = Chroma(
                    persist_directory=str(engine.vector_dir),
                    embedding_function=engine.embeddings,
                    collection_name="python_docs"
                )
            else:
                raise ValueError("No vector store found. Run ingestion first.")
        
        # Get context
        retriever = engine.vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build prompt
        prompt = prompt_template.format(context=context, question=query)
        
        # Measure time
        start_time = time.time()
        response = engine.llm.invoke(prompt)
        elapsed = time.time() - start_time
        
        # Extract answer text
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "query": query,
            "answer": answer,
            "response_time": elapsed,
            "answer_length": len(answer),
            "has_code": "```" in answer or "import" in answer.lower(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error evaluating prompt on query '{query}': {e}")
        return {
            "query": query,
            "answer": None,
            "response_time": None,
            "answer_length": None,
            "has_code": False,
            "success": False,
            "error": str(e)
        }


def evaluate_prompts():
    """
    Compare different prompt templates.
    
    Returns:
        DataFrame with comparison results
    """
    print("=" * 80)
    print("LLM EVALUATION - Comparing Prompt Templates")
    print("=" * 80)
    print(f"\nTest set size: {len(TEST_QUERIES)} queries")
    print(f"Prompt variants: {len(PROMPT_TEMPLATES)}\n")
    
    # Initialize engine
    logger.info("Initializing RAG engine...")
    engine = SimpleRAGEngine()
    
    all_results = []
    
    for prompt_name, prompt_template in PROMPT_TEMPLATES.items():
        logger.info(f"\nTesting prompt variant: {prompt_name}")
        print(f"\n📝 Testing '{prompt_name}' prompt...")
        
        for query in TEST_QUERIES:
            result = evaluate_prompt_variant(engine, prompt_template, query)
            result['prompt_variant'] = prompt_name
            all_results.append(result)
            
            if result['success']:
                logger.info(f"  ✓ {query[:50]}... ({result['response_time']:.2f}s, {result['answer_length']} chars)")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Summary by prompt variant
    summary = df[df['success']].groupby('prompt_variant').agg({
        'response_time': ['mean', 'std'],
        'answer_length': ['mean', 'std'],
        'has_code': 'sum',
        'success': 'count'
    }).round(2)
    
    print("\n" + "=" * 80)
    print("PROMPT COMPARISON SUMMARY")
    print("=" * 80)
    print(summary)
    
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "llm_prompt_comparison.csv", index=False)
    
    print(f"\n📊 Results saved to: evaluation_results/llm_prompt_comparison.csv")
    
    return df


def evaluate_models():
    """
    Compare different LLM models.
    
    Returns:
        DataFrame with model comparison results
    """
    print("\n" + "=" * 80)
    print("LLM EVALUATION - Comparing Models")
    print("=" * 80)
    
    models_to_test = [
        {"name": "gpt-4", "temperature": 0.0},
        {"name": "gpt-3.5-turbo", "temperature": 0.0},
        {"name": "gpt-4", "temperature": 0.7},
    ]
    
    # Use best prompt from previous evaluation (detailed)
    prompt_template = PROMPT_TEMPLATES["detailed"]
    
    all_results = []
    
    for model_config in models_to_test:
        model_name = model_config["name"]
        temperature = model_config["temperature"]
        
        logger.info(f"\nTesting model: {model_name} (temp={temperature})")
        print(f"\n🤖 Testing {model_name} (temperature={temperature})...")
        
        try:
            # Create engine with specific model
            llm = ChatOpenAI(model=model_name, temperature=temperature)
            engine = SimpleRAGEngine()
            engine.llm = llm
            
            for query in TEST_QUERIES:
                result = evaluate_prompt_variant(engine, prompt_template, query)
                result['model'] = model_name
                result['temperature'] = temperature
                result['config'] = f"{model_name}_temp{temperature}"
                all_results.append(result)
                
                if result['success']:
                    logger.info(f"  ✓ {query[:50]}... ({result['response_time']:.2f}s)")
                    
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) > 0:
        # Summary by model configuration
        summary = df[df['success']].groupby('config').agg({
            'response_time': ['mean', 'std'],
            'answer_length': ['mean', 'std'],
            'has_code': 'sum',
            'success': 'count'
        }).round(2)
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(summary)
        
        # Save results
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / "llm_model_comparison.csv", index=False)
        
        print(f"\n📊 Results saved to: evaluation_results/llm_model_comparison.csv")
    
    return df


def evaluate_temperature():
    """
    Compare different temperature settings for GPT-4.
    
    Returns:
        DataFrame with temperature comparison results
    """
    print("\n" + "=" * 80)
    print("LLM EVALUATION - Comparing Temperature Settings")
    print("=" * 80)
    
    temperatures = [0.0, 0.3, 0.7, 1.0]
    prompt_template = PROMPT_TEMPLATES["detailed"]
    
    all_results = []
    
    for temp in temperatures:
        logger.info(f"\nTesting temperature: {temp}")
        print(f"\n🌡️ Testing temperature={temp}...")
        
        try:
            llm = ChatOpenAI(model="gpt-4", temperature=temp)
            engine = SimpleRAGEngine()
            engine.llm = llm
            
            # Test on subset of queries
            for query in TEST_QUERIES[:5]:  # Use first 5 queries
                result = evaluate_prompt_variant(engine, prompt_template, query)
                result['temperature'] = temp
                all_results.append(result)
                
                if result['success']:
                    logger.info(f"  ✓ {query[:50]}... ({result['answer_length']} chars)")
                    
        except Exception as e:
            logger.error(f"Error testing temperature {temp}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) > 0:
        summary = df[df['success']].groupby('temperature').agg({
            'response_time': 'mean',
            'answer_length': 'mean',
            'success': 'count'
        }).round(2)
        
        print("\n" + "=" * 80)
        print("TEMPERATURE COMPARISON SUMMARY")
        print("=" * 80)
        print(summary)
        
        # Save results
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / "llm_temperature_comparison.csv", index=False)
        
        print(f"\n📊 Results saved to: evaluation_results/llm_temperature_comparison.csv")
    
    return df


def show_sample_outputs(df_prompts):
    """
    Show sample outputs for each prompt variant.
    
    Args:
        df_prompts: DataFrame with prompt comparison results
    """
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS BY PROMPT VARIANT")
    print("=" * 80)
    
    # Get first successful result for each prompt variant
    for prompt_variant in PROMPT_TEMPLATES.keys():
        samples = df_prompts[(df_prompts['prompt_variant'] == prompt_variant) & 
                            (df_prompts['success'] == True)]
        
        if len(samples) > 0:
            sample = samples.iloc[0]
            print(f"\n📝 Prompt Variant: {prompt_variant}")
            print(f"Query: {sample['query']}")
            print(f"Answer length: {sample['answer_length']} chars")
            print(f"Response time: {sample['response_time']:.2f}s")
            print(f"Answer preview: {sample['answer'][:200]}...")
            print("-" * 80)


def main():
    """Main LLM evaluation function."""
    try:
        # 1. Evaluate different prompts
        df_prompts = evaluate_prompts()
        
        # 2. Show sample outputs
        if len(df_prompts) > 0:
            show_sample_outputs(df_prompts)
        
        # 3. Evaluate different models
        df_models = evaluate_models()
        
        # 4. Evaluate temperature settings
        df_temp = evaluate_temperature()
        
        print("\n" + "=" * 80)
        print("✅ LLM EVALUATION COMPLETE")
        print("=" * 80)
        print("\n📊 Results Summary:")
        print("  - Prompt comparison: evaluation_results/llm_prompt_comparison.csv")
        print("  - Model comparison: evaluation_results/llm_model_comparison.csv")
        print("  - Temperature comparison: evaluation_results/llm_temperature_comparison.csv")
        
        print("\n💡 Recommendations:")
        
        # Find best prompt
        if len(df_prompts) > 0:
            best_prompt = df_prompts[df_prompts['success']].groupby('prompt_variant')['answer_length'].mean().idxmax()
            print(f"  ✅ Best prompt variant: {best_prompt}")
        
        # Find best model
        if len(df_models) > 0:
            best_model = df_models[df_models['success']].groupby('config')['response_time'].mean().idxmin()
            print(f"  ✅ Fastest model config: {best_model}")
        
        print("\nNext steps:")
        print("1. Review the CSV files for detailed analysis")
        print("2. Update RAG engine to use best configuration")
        print("3. Create LLM_EVALUATION.md with findings")
        
    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
