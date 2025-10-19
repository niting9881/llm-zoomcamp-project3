# RAG Evaluation FAQ: Comprehensive Guide to Metrics and Techniques

## Table of Contents
- [General Questions](#general-questions)
- [Non-LLM Based Metrics](#non-llm-based-metrics)
- [LLM-Based Metrics](#llm-based-metrics)
- [Hybrid Evaluation Approaches](#hybrid-evaluation-approaches)
- [Implementation and Best Practices](#implementation-and-best-practices)
- [Choosing the Right Metrics](#choosing-the-right-metrics)

---

## General Questions

### Q1: What is RAG evaluation and why is it important?

**Answer**: RAG (Retrieval-Augmented Generation) evaluation is the process of measuring the quality and effectiveness of a RAG system. It's important because:

- **Performance Measurement**: Understand how well your system retrieves relevant information and generates accurate answers
- **Optimization**: Identify bottlenecks and areas for improvement
- **Configuration Tuning**: Compare different models, prompts, and retrieval strategies
- **Production Monitoring**: Track system performance over time
- **Cost-Benefit Analysis**: Balance accuracy against computational costs

### Q2: What are the two main categories of RAG evaluation metrics?

**Answer**: 

1. **Non-LLM Based Metrics** (Traditional/Classical Metrics)
   - Don't require an LLM for evaluation
   - Fast, cheap, deterministic
   - Focus on retrieval quality
   - Examples: Hit Rate, MRR, Precision, Recall

2. **LLM-Based Metrics** (Modern/AI-Powered Metrics)
   - Use LLMs to evaluate quality
   - More nuanced, context-aware
   - Focus on answer quality and relevance
   - Examples: Faithfulness, Answer Relevance, Context Precision

### Q3: When should I use non-LLM vs LLM-based metrics?

**Answer**:

**Use Non-LLM Metrics When**:
- You need fast, cheap evaluation
- You have clear ground truth data
- You're optimizing retrieval components
- You need reproducible, deterministic results
- Budget is limited

**Use LLM-Based Metrics When**:
- You need semantic understanding
- Ground truth is hard to define
- You're evaluating generated text quality
- You need human-like judgment
- Accuracy is more important than cost

**Use Both When**:
- You want comprehensive evaluation
- You're optimizing the entire pipeline
- You need different perspectives on quality

---

## Non-LLM Based Metrics

### Q4: What is Hit Rate and how do I calculate it?

**Answer**: Hit Rate measures whether the correct document appears in the top-k retrieved results.

**Formula**:
```
Hit Rate = (Number of queries with at least one relevant doc in top-k) / (Total number of queries)
```

**Example**:
```python
def calculate_hit_rate(results, ground_truth, k=3):
    hits = 0
    for query_id, retrieved_docs in results.items():
        expected_docs = ground_truth[query_id]
        top_k = retrieved_docs[:k]
        if any(doc in expected_docs for doc in top_k):
            hits += 1
    return hits / len(results)
```

**Interpretation**:
- Range: 0.0 to 1.0 (0% to 100%)
- Higher is better
- Measures recall at position k
- Doesn't care about ranking order

**When to use**: When you want to know if relevant documents are being retrieved at all, regardless of their position.

### Q5: What is Mean Reciprocal Rank (MRR)?

**Answer**: MRR measures the rank position of the first relevant document.

**Formula**:
```
MRR = (1/Q) × Σ(1/rank_i)

where:
- Q = number of queries
- rank_i = position of first relevant document for query i
```

**Example**:
```python
def calculate_mrr(results, ground_truth):
    reciprocal_ranks = []
    for query_id, retrieved_docs in results.items():
        expected_docs = ground_truth[query_id]
        for i, doc in enumerate(retrieved_docs, start=1):
            if doc in expected_docs:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

**Interpretation**:
- Range: 0.0 to 1.0
- Higher is better
- MRR = 1.0: Perfect (relevant doc always at position 1)
- MRR = 0.5: On average, first relevant doc at position 2
- Sensitive to ranking quality

**When to use**: When the position of the first relevant document matters (e.g., users typically look at top results).

### Q6: What are Precision and Recall in RAG context?

**Answer**: 

**Precision**: Of the documents retrieved, how many are relevant?
```
Precision@k = (Number of relevant docs in top-k) / k
```

**Recall**: Of all relevant documents, how many were retrieved?
```
Recall@k = (Number of relevant docs in top-k) / (Total relevant docs)
```

**Example**:
```python
def calculate_precision_recall(retrieved_docs, relevant_docs, k):
    top_k = retrieved_docs[:k]
    relevant_in_top_k = [doc for doc in top_k if doc in relevant_docs]
    
    precision = len(relevant_in_top_k) / k
    recall = len(relevant_in_top_k) / len(relevant_docs)
    
    return precision, recall
```

**Trade-off**:
- High k → Higher recall, lower precision
- Low k → Lower recall, higher precision
- Use F1 score to balance both

**When to use**:
- Precision: When false positives are costly
- Recall: When missing relevant documents is costly
- F1: When you need balance

### Q7: What is NDCG (Normalized Discounted Cumulative Gain)?

**Answer**: NDCG measures ranking quality considering both relevance and position.

**Formula**:
```
DCG@k = Σ(relevance_i / log2(i + 1))
NDCG@k = DCG@k / IDCG@k

where:
- relevance_i = relevance score at position i
- IDCG = Ideal DCG (best possible ordering)
```

**Example**:
```python
import math

def calculate_ndcg(retrieved_docs, relevance_scores, k):
    dcg = sum(
        relevance_scores.get(doc, 0) / math.log2(i + 2)
        for i, doc in enumerate(retrieved_docs[:k])
    )
    
    # Calculate ideal DCG
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal_scores))
    
    return dcg / idcg if idcg > 0 else 0.0
```

**Interpretation**:
- Range: 0.0 to 1.0
- 1.0 = Perfect ranking
- Rewards putting highly relevant docs at top positions
- Penalizes poor ranking

**When to use**: When ranking order matters and you have graded relevance (not just binary).

### Q8: What is Context Recall (Non-LLM version)?

**Answer**: Non-LLM Context Recall measures what percentage of ground truth information appears in retrieved context.

**Formula**:
```
Context Recall = (Number of ground truth facts in retrieved context) / (Total ground truth facts)
```

**Example**:
```python
def calculate_context_recall(retrieved_context, ground_truth_facts):
    # Simple keyword matching approach
    facts_found = 0
    for fact in ground_truth_facts:
        if fact.lower() in retrieved_context.lower():
            facts_found += 1
    return facts_found / len(ground_truth_facts)
```

**Interpretation**:
- Range: 0.0 to 1.0
- Higher is better
- Measures completeness of retrieval
- Doesn't consider generated answer

**When to use**: When you want to evaluate retrieval quality independently of generation quality.

### Q9: How do I measure retrieval latency?

**Answer**: Latency measures response time of retrieval and generation.

**Metrics to Track**:
```python
import time

def measure_rag_latency(query):
    start_time = time.time()
    
    # Retrieval phase
    retrieval_start = time.time()
    docs = retriever.retrieve(query)
    retrieval_time = time.time() - retrieval_start
    
    # Generation phase
    generation_start = time.time()
    answer = llm.generate(query, docs)
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start_time
    
    return {
        'total_latency': total_time,
        'retrieval_latency': retrieval_time,
        'generation_latency': generation_time,
        'retrieval_percentage': (retrieval_time / total_time) * 100
    }
```

**Key Metrics**:
- **Total Latency**: End-to-end response time
- **Retrieval Latency**: Time to retrieve documents
- **Generation Latency**: Time to generate answer
- **P50, P95, P99**: Percentile latencies

**When to use**: Always! Latency is critical for user experience.

---

## LLM-Based Metrics

### Q10: What is Faithfulness and how is it evaluated?

**Answer**: Faithfulness measures whether the generated answer is factually consistent with the retrieved context (no hallucinations).

**Evaluation Method**:
```python
def evaluate_faithfulness(answer, context, llm):
    prompt = f"""
Given the following context and answer, determine if the answer is 
faithful to the context (contains no hallucinations).

Context: {context}

Answer: {answer}

Is the answer faithful to the context? Answer with:
- "YES" if all claims in the answer are supported by the context
- "NO" if any claim is not supported or contradicts the context

Explanation: [Your reasoning]

Verdict: [YES/NO]
"""
    
    response = llm.invoke(prompt)
    verdict = extract_verdict(response)  # Parse YES/NO
    
    return {
        'is_faithful': verdict == 'YES',
        'score': 1.0 if verdict == 'YES' else 0.0,
        'explanation': extract_explanation(response)
    }
```

**Interpretation**:
- Binary: Faithful (1.0) or Not Faithful (0.0)
- Or graded: Score from 0.0 to 1.0
- Lower = More hallucinations
- Higher = Better adherence to context

**When to use**: When accuracy is critical and hallucinations must be minimized (e.g., medical, legal, financial domains).

### Q11: What is Answer Relevance?

**Answer**: Answer Relevance measures how well the generated answer addresses the original question.

**Evaluation Method**:
```python
def evaluate_answer_relevance(question, answer, llm):
    prompt = f"""
Rate how relevant the following answer is to the question on a scale of 1-5:

Question: {question}

Answer: {answer}

Consider:
- Does it directly address the question?
- Is it complete?
- Is it concise (not too verbose)?

Rating (1-5): [Your score]
Explanation: [Your reasoning]
"""
    
    response = llm.invoke(prompt)
    score = extract_score(response)  # Parse 1-5
    
    return {
        'relevance_score': score / 5.0,  # Normalize to 0-1
        'explanation': extract_explanation(response)
    }
```

**Scoring Guide**:
- **5/5**: Perfect - Directly answers question, complete, concise
- **4/5**: Good - Answers question with minor issues
- **3/5**: Acceptable - Partially answers or too verbose
- **2/5**: Poor - Tangentially related
- **1/5**: Irrelevant - Doesn't answer question

**When to use**: To ensure generated answers actually address user queries.

### Q12: What is Context Precision (LLM-based)?

**Answer**: Context Precision measures whether all retrieved documents are relevant to answering the question.

**Evaluation Method**:
```python
def evaluate_context_precision(question, retrieved_docs, llm):
    relevant_count = 0
    
    for i, doc in enumerate(retrieved_docs):
        prompt = f"""
Is the following document relevant for answering the question?

Question: {question}

Document: {doc}

Answer with YES or NO and explain why.

Verdict: [YES/NO]
"""
        
        response = llm.invoke(prompt)
        if extract_verdict(response) == 'YES':
            relevant_count += 1
    
    precision = relevant_count / len(retrieved_docs)
    return {
        'context_precision': precision,
        'relevant_docs': relevant_count,
        'total_docs': len(retrieved_docs)
    }
```

**Interpretation**:
- Range: 0.0 to 1.0
- 1.0 = All retrieved docs are relevant
- Lower scores = Noise in context
- Helps identify if retrieval is too broad

**When to use**: When you want to minimize irrelevant context (reduces cost and improves generation quality).

### Q13: What is Context Relevancy?

**Answer**: Context Relevancy measures how much of the retrieved context is actually needed to answer the question.

**Evaluation Method**:
```python
def evaluate_context_relevancy(question, answer, context, llm):
    prompt = f"""
Analyze the retrieved context and identify which parts are relevant 
for answering the question.

Question: {question}

Answer: {answer}

Context: {context}

What percentage of the context is actually relevant and used in the answer?

Percentage (0-100): [Your estimate]
Explanation: [Which parts are relevant/irrelevant]
"""
    
    response = llm.invoke(prompt)
    percentage = extract_percentage(response)
    
    return {
        'context_relevancy': percentage / 100.0,
        'explanation': extract_explanation(response)
    }
```

**Interpretation**:
- Range: 0.0 to 1.0
- Higher = More focused retrieval
- Lower = Too much irrelevant context
- Indicates if you're retrieving too much

**When to use**: To optimize retrieval (reduce k or improve query formulation).

### Q14: What is Answer Correctness?

**Answer**: Answer Correctness measures factual accuracy of the generated answer against ground truth.

**Evaluation Method**:
```python
def evaluate_answer_correctness(question, generated_answer, ground_truth, llm):
    prompt = f"""
Compare the generated answer with the ground truth answer and rate 
the correctness on a scale of 1-5.

Question: {question}

Generated Answer: {generated_answer}

Ground Truth: {ground_truth}

Consider:
- Factual accuracy
- Completeness
- Semantic similarity

Rating (1-5): [Your score]
Explanation: [Your reasoning]
"""
    
    response = llm.invoke(prompt)
    score = extract_score(response)
    
    return {
        'correctness_score': score / 5.0,
        'explanation': extract_explanation(response)
    }
```

**Interpretation**:
- Range: 0.0 to 1.0
- 1.0 = Perfectly correct
- Can combine with semantic similarity scores
- Accounts for paraphrasing

**When to use**: When you have ground truth answers and want to measure accuracy.

### Q15: What is Answer Semantic Similarity?

**Answer**: Answer Semantic Similarity measures how semantically similar the generated answer is to a reference answer, accounting for paraphrasing.

**Evaluation Method**:
```python
from sentence_transformers import SentenceTransformer, util

def evaluate_semantic_similarity(generated_answer, reference_answer):
    # Using embeddings for semantic comparison
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embedding1 = model.encode(generated_answer, convert_to_tensor=True)
    embedding2 = model.encode(reference_answer, convert_to_tensor=True)
    
    similarity = util.cos_sim(embedding1, embedding2).item()
    
    return {
        'semantic_similarity': similarity,
        'interpretation': 'High' if similarity > 0.8 else 'Medium' if similarity > 0.5 else 'Low'
    }
```

**Interpretation**:
- Range: -1.0 to 1.0 (typically 0.0 to 1.0)
- > 0.8: Very similar (different wording, same meaning)
- 0.5-0.8: Moderately similar
- < 0.5: Different meaning

**When to use**: When you want to allow flexible wording but measure semantic equivalence.

---

## Hybrid Evaluation Approaches

### Q16: What is the RAGAS framework?

**Answer**: RAGAS (Retrieval Augmented Generation Assessment) is a comprehensive framework combining multiple metrics.

**Key Metrics**:
1. **Context Precision**: Are retrieved docs relevant?
2. **Context Recall**: Is all necessary info retrieved?
3. **Faithfulness**: Is answer grounded in context?
4. **Answer Relevance**: Does answer address question?

**Implementation**:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

def evaluate_with_ragas(dataset):
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    return results
```

**When to use**: For comprehensive, standardized RAG evaluation with both retrieval and generation quality metrics.

### Q17: How do I combine multiple metrics effectively?

**Answer**: Use a weighted scoring system based on your priorities.

**Example Framework**:
```python
def calculate_composite_score(metrics, weights=None):
    """
    Combine multiple metrics into a single score
    
    Args:
        metrics: dict of metric_name -> score (0.0 to 1.0)
        weights: dict of metric_name -> weight (sums to 1.0)
    """
    if weights is None:
        # Default equal weights
        weights = {k: 1.0/len(metrics) for k in metrics.keys()}
    
    composite = sum(metrics[k] * weights[k] for k in metrics.keys())
    
    return {
        'composite_score': composite,
        'individual_scores': metrics,
        'weights': weights
    }

# Example usage
metrics = {
    'hit_rate': 0.85,
    'mrr': 0.72,
    'faithfulness': 0.90,
    'answer_relevance': 0.88,
    'latency_score': 0.95  # Normalized (1 - normalized_latency)
}

weights = {
    'hit_rate': 0.20,
    'mrr': 0.15,
    'faithfulness': 0.30,  # High weight for accuracy
    'answer_relevance': 0.25,
    'latency_score': 0.10
}

score = calculate_composite_score(metrics, weights)
print(f"Composite Score: {score['composite_score']:.2f}")
```

**When to use**: When you need a single number to compare different RAG configurations.

### Q18: What is the difference between online and offline evaluation?

**Answer**:

**Offline Evaluation** (Pre-Production):
- Test on static dataset
- Automated metrics
- Fast iteration
- Cheaper
- Example: Hit Rate, MRR, NDCG
- Use during development

**Online Evaluation** (Production):
- Real user interactions
- Human feedback
- A/B testing
- More expensive but realistic
- Example: User satisfaction, click-through rate, time to answer
- Use for monitoring

**Best Practice**: Use offline for development, online for validation and monitoring.

```python
class RAGEvaluator:
    def offline_eval(self, test_set):
        """Evaluate on static test set"""
        return {
            'hit_rate': self.calculate_hit_rate(test_set),
            'mrr': self.calculate_mrr(test_set),
            'faithfulness': self.calculate_faithfulness(test_set)
        }
    
    def online_eval(self, interactions):
        """Evaluate from production data"""
        return {
            'user_satisfaction': self.calculate_satisfaction(interactions),
            'thumbs_up_rate': self.calculate_thumbs_up_rate(interactions),
            'avg_session_length': self.calculate_session_length(interactions)
        }
```

---

## Implementation and Best Practices

### Q19: How many test queries do I need for reliable evaluation?

**Answer**: 

**Minimum Recommendations**:
- **Development**: 15-30 queries per category
- **Production**: 100-500 queries total
- **Comprehensive**: 1000+ queries

**Guidelines**:
```python
def create_test_set(categories, queries_per_category=20):
    """
    Create balanced test set
    
    Args:
        categories: List of query types/categories
        queries_per_category: Target queries per category
    """
    test_set = []
    
    for category in categories:
        # Ensure diversity within category
        queries = generate_diverse_queries(
            category=category,
            count=queries_per_category,
            difficulty_levels=['easy', 'medium', 'hard']
        )
        test_set.extend(queries)
    
    return test_set

# Example
categories = [
    'factual',
    'procedural', 
    'comparative',
    'troubleshooting',
    'explanation'
]

test_set = create_test_set(categories, queries_per_category=20)
print(f"Total test queries: {len(test_set)}")  # 100 queries
```

**Quality over Quantity**:
- Diverse query types
- Range of difficulty levels
- Cover edge cases
- Balanced categories

### Q20: How do I create ground truth data?

**Answer**:

**Methods**:

1. **Manual Annotation** (Most Accurate):
```python
def create_ground_truth_manual():
    ground_truth = {
        'query_1': {
            'question': 'What is pandas?',
            'expected_docs': ['pandas_doc', 'numpy_doc'],
            'reference_answer': 'pandas is a Python library for data manipulation...',
            'relevance_scores': {'pandas_doc': 5, 'numpy_doc': 3}
        }
    }
    return ground_truth
```

2. **LLM-Assisted Generation**:
```python
def generate_ground_truth_with_llm(documents, llm):
    ground_truth = []
    
    for doc in documents:
        # Generate questions from document
        questions = llm.generate_questions(doc)
        
        for q in questions:
            ground_truth.append({
                'question': q,
                'expected_docs': [doc.id],
                'reference_answer': extract_answer(doc, q)
            })
    
    return ground_truth
```

3. **Production Data Mining**:
```python
def extract_ground_truth_from_logs(user_logs):
    """Use queries with positive feedback as ground truth"""
    ground_truth = []
    
    for log in user_logs:
        if log['user_feedback'] == 'positive':
            ground_truth.append({
                'question': log['query'],
                'reference_answer': log['answer'],
                'implicit_relevance': True
            })
    
    return ground_truth
```

### Q21: What tools and libraries are available for RAG evaluation?

**Answer**:

**Popular Frameworks**:

1. **RAGAS** (Comprehensive):
```bash
pip install ragas
```
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
```

2. **LangChain Evaluators**:
```python
from langchain.evaluation import (
    load_evaluator,
    QAEvalChain
)

evaluator = load_evaluator("qa")
results = evaluator.evaluate_strings(
    prediction=answer,
    input=question,
    reference=ground_truth
)
```

3. **TruLens** (Monitoring):
```python
from trulens_eval import TruChain, Feedback

# Track and evaluate in production
tru_chain = TruChain(chain, feedbacks=[...])
```

4. **Custom Implementation**:
```python
class RAGEvaluator:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
    
    def evaluate_all(self, test_set):
        results = {
            'hit_rate': self.hit_rate(test_set),
            'mrr': self.mrr(test_set),
            'faithfulness': self.faithfulness(test_set),
            'answer_relevance': self.answer_relevance(test_set)
        }
        return results
```

### Q22: How often should I run evaluations?

**Answer**:

**Development Phase**:
- After each code change
- Before merging to main
- Daily automated runs

**Production Phase**:
- Continuous: Real-time metrics (latency, error rate)
- Hourly: Automated quality checks on sample
- Daily: Full evaluation on test set
- Weekly: Comprehensive analysis with reports
- Monthly: Deep dive with human review

**Automation Example**:
```python
import schedule

def run_evaluation_pipeline():
    # Load test set
    test_set = load_test_set()
    
    # Run evaluations
    results = evaluate_rag_system(test_set)
    
    # Store results
    save_results(results, timestamp=datetime.now())
    
    # Alert if degradation
    if results['composite_score'] < threshold:
        send_alert("RAG quality degradation detected!")
    
    # Generate report
    generate_report(results)

# Schedule evaluations
schedule.every().day.at("02:00").do(run_evaluation_pipeline)
schedule.every().hour.do(run_quick_evaluation)
```

---

## Choosing the Right Metrics

### Q23: Which metrics should I start with?

**Answer**:

**Minimum Viable Evaluation** (Start Here):
1. **Hit Rate @3**: Are relevant docs being retrieved?
2. **Latency**: Is the system fast enough?
3. **Faithfulness**: Are answers accurate?
4. **User Feedback**: Are users satisfied?

**Implementation**:
```python
def minimum_viable_evaluation(rag_system, test_set):
    """Quick evaluation to validate basic functionality"""
    return {
        'hit_rate': evaluate_hit_rate(rag_system, test_set, k=3),
        'avg_latency': measure_latency(rag_system, test_set),
        'faithfulness': evaluate_faithfulness(rag_system, test_set),
        'user_satisfaction': get_user_feedback_score()
    }
```

**Progressive Enhancement**:
- **Phase 1**: Hit Rate, Latency → Validate retrieval works
- **Phase 2**: + MRR, Faithfulness → Optimize ranking and accuracy
- **Phase 3**: + Answer Relevance, Context Precision → Fine-tune generation
- **Phase 4**: + Full RAGAS suite → Comprehensive quality

### Q24: How do I evaluate RAG for different use cases?

**Answer**:

**Customer Support**:
- **Priority**: Answer Relevance, Latency
- **Metrics**: Resolution rate, Time to answer, User satisfaction
```python
support_metrics = {
    'answer_relevance': 0.35,  # Must answer the question
    'latency': 0.25,           # Fast response critical
    'faithfulness': 0.25,      # Accurate information
    'context_recall': 0.15     # Complete information
}
```

**Medical/Legal (High Stakes)**:
- **Priority**: Faithfulness, Context Precision
- **Metrics**: Zero hallucinations, Source attribution
```python
high_stakes_metrics = {
    'faithfulness': 0.50,       # No hallucinations
    'context_precision': 0.30,  # Only relevant sources
    'answer_correctness': 0.20  # Factually accurate
}
```

**Casual Q&A (Search)**:
- **Priority**: Hit Rate, Latency
- **Metrics**: Relevance, Speed
```python
search_metrics = {
    'hit_rate': 0.40,          # Find relevant docs
    'latency': 0.30,           # Fast results
    'answer_relevance': 0.30   # Useful answers
}
```

**Research/Analysis**:
- **Priority**: Context Recall, Comprehensiveness
- **Metrics**: Completeness, Depth
```python
research_metrics = {
    'context_recall': 0.40,     # All relevant info
    'answer_completeness': 0.30, # Comprehensive
    'context_precision': 0.20,  # Quality sources
    'citations': 0.10           # Proper attribution
}
```

### Q25: What are the common pitfalls in RAG evaluation?

**Answer**:

**Pitfall 1: Only Evaluating One Component**
```python
# ❌ Wrong: Only evaluate retrieval
hit_rate = evaluate_retrieval(system)

# ✅ Correct: Evaluate end-to-end
evaluation = {
    'retrieval': evaluate_retrieval(system),
    'generation': evaluate_generation(system),
    'latency': measure_latency(system),
    'user_experience': get_user_feedback(system)
}
```

**Pitfall 2: Insufficient Test Set**
```python
# ❌ Wrong: Too few, not diverse
test_set = ['What is X?', 'What is Y?']

# ✅ Correct: Diverse, comprehensive
test_set = create_test_set(
    categories=['factual', 'procedural', 'comparative'],
    difficulty=['easy', 'medium', 'hard'],
    min_per_category=20
)
```

**Pitfall 3: Not Testing Edge Cases**
```python
# ✅ Include edge cases
edge_cases = [
    'Ambiguous queries',
    'Multi-hop reasoning required',
    'Out-of-domain questions',
    'Questions with no answer in corpus',
    'Very long queries',
    'Queries with typos'
]
```

**Pitfall 4: Ignoring Production Metrics**
```python
# ❌ Wrong: Only offline eval
offline_score = evaluate_on_test_set()

# ✅ Correct: Monitor production
monitor_production_metrics(
    user_satisfaction,
    query_distribution,
    failure_rates,
    latency_percentiles
)
```

**Pitfall 5: Over-relying on Single Metric**
```python
# ❌ Wrong: Optimize only for hit rate
optimize_for_hit_rate()  # Might retrieve too many docs

# ✅ Correct: Balance multiple objectives
optimize_multi_objective(
    hit_rate_weight=0.3,
    latency_weight=0.3,
    faithfulness_weight=0.4
)
```

---

## Summary Cheat Sheet

### Quick Reference: When to Use Which Metric

| Metric | Type | What It Measures | Best For | Cost |
|--------|------|------------------|----------|------|
| **Hit Rate** | Non-LLM | Is relevant doc in top-k? | Quick retrieval check | Free |
| **MRR** | Non-LLM | Position of first relevant doc | Ranking quality | Free |
| **NDCG** | Non-LLM | Ranking quality (graded) | Advanced ranking | Free |
| **Precision/Recall** | Non-LLM | Retrieval accuracy | Balanced evaluation | Free |
| **Latency** | Non-LLM | Response speed | UX optimization | Free |
| **Faithfulness** | LLM | No hallucinations | Accuracy critical | $$ |
| **Answer Relevance** | LLM | Addresses question | Quality assurance | $$ |
| **Context Precision** | LLM | Relevant docs only | Reduce noise | $$ |
| **Context Recall** | LLM | All info retrieved | Completeness | $$ |
| **Answer Correctness** | LLM | Factual accuracy | Ground truth comparison | $$ |

### Recommended Evaluation Pipeline

```python
def comprehensive_rag_evaluation(rag_system, test_set):
    """Complete evaluation pipeline"""
    
    # Phase 1: Fast metrics (no cost)
    fast_metrics = {
        'hit_rate_k3': calculate_hit_rate(rag_system, test_set, k=3),
        'hit_rate_k5': calculate_hit_rate(rag_system, test_set, k=5),
        'mrr': calculate_mrr(rag_system, test_set),
        'avg_latency': measure_latency(rag_system, test_set),
        'p95_latency': measure_latency_p95(rag_system, test_set)
    }
    
    # Phase 2: LLM-based metrics (sample to reduce cost)
    sample = random.sample(test_set, min(50, len(test_set)))
    llm_metrics = {
        'faithfulness': evaluate_faithfulness(rag_system, sample),
        'answer_relevance': evaluate_relevance(rag_system, sample),
        'context_precision': evaluate_context_precision(rag_system, sample)
    }
    
    # Phase 3: User feedback (if available)
    user_metrics = {
        'thumbs_up_rate': get_thumbs_up_rate(),
        'avg_rating': get_average_rating()
    }
    
    # Combine all metrics
    return {
        **fast_metrics,
        **llm_metrics,
        **user_metrics,
        'composite_score': calculate_composite_score({**fast_metrics, **llm_metrics})
    }
```

---

## Additional Resources

### Papers and Articles
- RAGAS: Automated Evaluation of RAG
- Evaluating RAG Systems (OpenAI Cookbook)
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

### Tools and Frameworks
- **RAGAS**: https://github.com/explodinggradients/ragas
- **LangChain Evaluation**: https://python.langchain.com/docs/guides/evaluation
- **TruLens**: https://www.trulens.org/
- **DeepEval**: https://github.com/confident-ai/deepeval

### Best Practices
1. Start simple (Hit Rate + Latency)
2. Add complexity progressively
3. Always evaluate end-to-end
4. Monitor in production
5. Balance cost vs accuracy
6. Use multiple metrics
7. Test edge cases
8. Automate evaluation
9. Track over time
10. Get human feedback

---

*Last Updated: October 19, 2025*  
*Based on: LLM Zoomcamp 2025, RAG Best Practices*
