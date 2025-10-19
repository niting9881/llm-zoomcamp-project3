"""
Simplified RAG Engine - Built from scratch with working components
No advanced features - just basic retrieval and generation
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SimpleRAGEngine:
    """Simplified RAG engine with only essential features that work."""
    
    def __init__(self):
        """Initialize the RAG engine with working components only."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=self.api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=self.api_key
        )
        
        # Vector store path
        self.vector_dir = Path("data/processed/vectorstore")
        self.vector_store = None
        
        logger.info("SimpleRAGEngine initialized successfully")
    
    def ingest_docs(self, docs_dir: Path = Path("data/raw")) -> None:
        """
        Ingest documents from the data directory.
        Simplified version - just reads text files and creates embeddings.
        """
        logger.info(f"Ingesting documents from {docs_dir}")
        
        documents = []
        
        # Read Python docs
        python_docs_dir = docs_dir / "python_docs" / "3.11"
        if python_docs_dir.exists():
            for html_file in list(python_docs_dir.glob("*.html"))[:20]:  # Limit to 20 for testing
                try:
                    content = html_file.read_text(encoding='utf-8', errors='ignore')
                    # Simple extraction - just get first 2000 characters
                    content = content[:2000]
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(html_file.name),
                            "type": "python_doc"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to read {html_file}: {e}")
        
        # Read PyPI packages
        pypi_dir = docs_dir / "pypi_packages"
        if pypi_dir.exists():
            for readme_file in pypi_dir.glob("*_readme.md"):
                try:
                    content = readme_file.read_text(encoding='utf-8', errors='ignore')
                    # Limit content size
                    content = content[:2000]
                    
                    package_name = readme_file.stem.replace('_readme', '')
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": package_name,
                            "type": "pypi_package"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to read {readme_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        if documents:
            # Create vector store
            self.vector_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.vector_dir),
                collection_name="python_docs"
            )
            logger.info(f"Created vector store with {len(documents)} documents")
        else:
            logger.warning("No documents found to ingest")
    
    def query(self, query_text: str, k: int = 3) -> str:
        """
        Query the RAG system with a simple question.
        Returns an answer based on retrieved documents.
        """
        logger.info(f"Processing query: {query_text}")
        
        # Load vector store if not already loaded
        if self.vector_store is None:
            if self.vector_dir.exists():
                logger.info("Loading existing vector store...")
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_dir),
                    embedding_function=self.embeddings,
                    collection_name="python_docs"
                )
            else:
                raise ValueError("No vector store found. Run ingest_docs() first.")
        
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query_text)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Create context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        # Create prompt
        prompt = f"""You are a helpful Python documentation assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query_text}

Answer (be concise and helpful):"""
        
        # Get answer from LLM
        response = self.llm.invoke(prompt)
        answer = response.content
        
        logger.info("Generated answer successfully")
        
        # Return answer with sources
        sources = [doc.metadata.get('source', 'unknown') for doc in retrieved_docs]
        result = f"{answer}\n\nSources: {', '.join(sources)}"
        
        return result


# Test the simplified RAG engine
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Testing Simplified RAG Engine")
    print("=" * 80)
    
    # Initialize
    print("\n[1] Initializing RAG engine...")
    rag = SimpleRAGEngine()
    print("✓ Initialized successfully")
    
    # Ingest documents
    print("\n[2] Ingesting documents...")
    rag.ingest_docs()
    print("✓ Documents ingested")
    
    # Test queries
    test_queries = [
        "What is BeautifulSoup?",
        "How do I use pandas?",
        "What is NumPy?"
    ]
    
    for query in test_queries:
        print(f"\n[QUERY] {query}")
        print("-" * 80)
        answer = rag.query(query)
        print(answer)
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)
