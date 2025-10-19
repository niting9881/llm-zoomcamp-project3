"""
Test each component individually to identify what's working and what's not.
Run this script to verify each piece before building the full RAG system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("COMPONENT TESTING - RAG System")
print("=" * 80)

# Test 1: OpenAI API Key
print("\n[TEST 1] Checking OpenAI API Key...")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-5:]}")
else:
    print("✗ API Key NOT found in environment")
    exit(1)

# Test 2: Import LangChain components
print("\n[TEST 2] Importing LangChain components...")
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    print("✓ All LangChain imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 3: Initialize ChatOpenAI
print("\n[TEST 3] Initializing ChatOpenAI...")
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    print("✓ ChatOpenAI initialized")
except Exception as e:
    print(f"✗ ChatOpenAI initialization failed: {e}")
    exit(1)

# Test 4: Test LLM with simple query
print("\n[TEST 4] Testing LLM with simple query...")
try:
    response = llm.invoke("Say 'Hello, I am working!'")
    print(f"✓ LLM Response: {response.content}")
except Exception as e:
    print(f"✗ LLM query failed: {e}")
    exit(1)

# Test 5: Initialize OpenAI Embeddings
print("\n[TEST 5] Initializing OpenAI Embeddings...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    print("✓ Embeddings initialized")
except Exception as e:
    print(f"✗ Embeddings initialization failed: {e}")
    exit(1)

# Test 6: Test embedding generation
print("\n[TEST 6] Testing embedding generation...")
try:
    test_embedding = embeddings.embed_query("test document")
    print(f"✓ Embedding generated successfully (dimension: {len(test_embedding)})")
except Exception as e:
    print(f"✗ Embedding generation failed: {e}")
    exit(1)

# Test 7: Create simple ChromaDB
print("\n[TEST 7] Creating ChromaDB vector store...")
try:
    from langchain_core.documents import Document
    
    # Create sample documents
    docs = [
        Document(page_content="BeautifulSoup is a Python library for web scraping.", metadata={"source": "test1"}),
        Document(page_content="Pandas is a data manipulation library in Python.", metadata={"source": "test2"}),
        Document(page_content="NumPy provides support for large arrays and matrices.", metadata={"source": "test3"})
    ]
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="test_collection"
    )
    print("✓ ChromaDB vector store created")
except Exception as e:
    print(f"✗ ChromaDB creation failed: {e}")
    exit(1)

# Test 8: Test retrieval
print("\n[TEST 8] Testing document retrieval...")
try:
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("What is BeautifulSoup?")
    print(f"✓ Retrieved {len(results)} documents")
    for i, doc in enumerate(results):
        print(f"  Doc {i+1}: {doc.page_content[:50]}...")
except Exception as e:
    print(f"✗ Retrieval failed: {e}")
    exit(1)

# Test 9: Test simple RAG query
print("\n[TEST 9] Testing simple RAG query...")
try:
    # Simple RAG without fancy features
    query = "What is BeautifulSoup?"
    retrieved_docs = retriever.invoke(query)
    
    # Create context from retrieved docs
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    # Get answer
    answer = llm.invoke(prompt)
    print(f"✓ RAG Query successful!")
    print(f"  Question: {query}")
    print(f"  Answer: {answer.content}")
except Exception as e:
    print(f"✗ RAG query failed: {e}")
    exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
