from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
from bs4 import BeautifulSoup
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    """Modern RAG Engine using LangChain 0.1+ APIs"""
    
    def __init__(self, docs_dir="data/raw", vector_dir="data/processed/vectorstore"):
        self.docs_dir = Path(docs_dir)
        self.vector_dir = vector_dir
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        self.vector_store = None
        self.bm25 = None
        self.corpus = []
        self.corpus_metadatas = []
        # Use cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def ingest_docs(self, files=None):
        """Ingest documents from files and create vector store"""
        logger.info("Starting document ingestion...")
        docs = []
        metadatas = []
        
        if files is None:
            files = list(self.docs_dir.rglob("*.md")) + list(self.docs_dir.rglob("*.html"))
        
        logger.info(f"Found {len(files)} files to process")
        
        for file in files:
            try:
                if str(file).endswith(".md"):
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks = self.text_splitter.split_text(content)
                        docs.extend(chunks)
                        metadatas.extend([{"source": str(file), "type": "markdown"} for _ in chunks])
                elif str(file).endswith(".html"):
                    with open(file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                        soup = BeautifulSoup(html_content, "html.parser")
                        text = soup.get_text(separator="\n", strip=True)
                        chunks = self.text_splitter.split_text(text)
                        docs.extend(chunks)
                        metadatas.extend([{"source": str(file), "type": "html"} for _ in chunks])
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue
        
        if docs:
            logger.info(f"Creating vector store with {len(docs)} chunks")
            self.vector_store = Chroma.from_texts(
                docs, 
                self.embeddings, 
                metadatas=metadatas, 
                persist_directory=str(self.vector_dir)
            )
            self.corpus = docs
            self.corpus_metadatas = metadatas
            tokenized_corpus = [doc.split() for doc in docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("Document ingestion complete")
        else:
            logger.warning("No documents were ingested")

    def query(self, query_text, k=5):
        """Query the RAG system with hybrid search and reranking"""
        logger.info(f"Processing query: {query_text}")
        
        # Step 1: Query rewriting using LLM
        rewrite_prompt = f"Rewrite this query to be more specific and detailed for better document retrieval. Only return the rewritten query, nothing else:\n\nOriginal query: {query_text}\n\nRewritten query:"
        
        try:
            rewritten_query = self.llm.invoke(rewrite_prompt).content.strip()
            logger.info(f"Rewritten query: {rewritten_query}")
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}, using original query")
            rewritten_query = query_text

        # Step 2: Vector search
        if not self.vector_store:
            logger.info("Loading existing vector store")
            self.vector_store = Chroma(
                persist_directory=str(self.vector_dir), 
                embedding_function=self.embeddings
            )
        
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            vector_docs = retriever.invoke(rewritten_query)
            logger.info(f"Retrieved {len(vector_docs)} documents from vector search")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            vector_docs = []

        # Step 3: BM25 keyword search
        bm25_docs = []
        if self.bm25 and len(self.corpus) > 0:
            try:
                tokenized_query = rewritten_query.split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_indices = sorted(
                    range(len(bm25_scores)), 
                    key=lambda i: bm25_scores[i], 
                    reverse=True
                )[:k]
                
                for i in bm25_indices:
                    doc = Document(
                        page_content=self.corpus[i],
                        metadata=self.corpus_metadatas[i]
                    )
                    bm25_docs.append(doc)
                
                logger.info(f"Retrieved {len(bm25_docs)} documents from BM25 search")
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
                bm25_docs = []

        # Step 4: Combine and deduplicate results
        combined_docs = vector_docs + bm25_docs
        seen_content = set()
        unique_docs = []
        for doc in combined_docs:
            content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        logger.info(f"Combined to {len(unique_docs)} unique documents")

        # Step 5: Rerank using cross-encoder
        if len(unique_docs) > 0:
            try:
                pairs = [[rewritten_query, doc.page_content] for doc in unique_docs]
                scores = self.reranker.predict(pairs)
                reranked_indices = sorted(
                    range(len(scores)), 
                    key=lambda i: scores[i], 
                    reverse=True
                )[:k]
                reranked_docs = [unique_docs[i] for i in reranked_indices]
                logger.info(f"Reranked to top {len(reranked_docs)} documents")
            except Exception as e:
                logger.error(f"Reranking failed: {e}, using original order")
                reranked_docs = unique_docs[:k]
        else:
            reranked_docs = []

        # Step 6: Generate answer using LLM
        if len(reranked_docs) > 0:
            context = "\n\n---\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
                for doc in reranked_docs[:3]
            ])
            
            prompt = f"""You are a helpful assistant that answers questions about Python programming and libraries based on official documentation.

Context from documentation:
{context}

Question: {query_text}

Instructions:
- Provide a clear, accurate answer based on the context above
- If the context doesn't contain enough information, say so
- Include specific examples or code snippets if relevant
- Cite sources when appropriate

Answer:"""
            
            try:
                answer = self.llm.invoke(prompt).content
                logger.info("Generated answer successfully")
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                answer = f"Error generating answer: {str(e)}"
        else:
            answer = "I couldn't find relevant information to answer your question. Please try rephrasing or asking something else."
        
        # Format results
        sources = [
            {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "unknown")
            } 
            for doc in reranked_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "query_original": query_text,
            "query_rewritten": rewritten_query,
            "num_sources": len(sources)
        }
