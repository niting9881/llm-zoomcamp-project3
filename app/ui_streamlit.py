import streamlit as st
import requests
import time
import json

# Page configuration
st.set_page_config(
    page_title="Python RAG Documentation Helper", 
    layout="wide",
    page_icon="🐍"
)

# Title and description
st.title("🐍 Python Documentation RAG System")
st.markdown("""
### 🤖 AI-Powered Python Documentation Assistant

**What this tool does:**
- 📚 **Searches official Python documentation** (versions 3.14, 3.12, 3.11)
- 📦 **Searches PyPI package documentation** (BeautifulSoup4, LangChain, Pandas, NumPy)
- 🧠 **Uses GPT-4** to provide accurate, context-aware answers
- 🔍 **Shows source references** so you can verify the information
- ⚡ **Fast responses** powered by vector similarity search and RAG

**How to use:**
1. Enter your question about Python or a specific library
2. Optionally filter by package name
3. Get an AI-generated answer with source citations
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system:
    - Searches through Python documentation
    - Retrieves relevant context
    - Generates accurate answers using GPT-4
    - Provides source attribution
    """)
    
    st.header("📚 Available Sources")
    st.markdown("""
    **Python Versions:**
    - Python 3.14 Documentation
    - Python 3.12 Documentation
    - Python 3.11 Documentation
    
    **PyPI Packages:**
    - BeautifulSoup4
    - LangChain
    - Pandas
    - NumPy
    """)
    
    st.header("🔧 API Status")
    try:
        health_check = requests.get("http://127.0.0.1:8000/", timeout=2)
        st.success("✅ API Connected")
    except:
        st.error("❌ API Disconnected")

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔍 Enter your question:", 
        placeholder="e.g., What is BeautifulSoup? How do I use pandas DataFrames?",
        help="Ask any question about Python libraries or language features"
    )

with col2:
    package = st.text_input(
        "📦 Filter by package (optional):", 
        placeholder="e.g., pandas",
        help="Leave empty to search all sources"
    )

# Search button
if st.button("🚀 Search", type="primary", use_container_width=True):
    if not query:
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("🔎 Searching documentation and generating answer..."):
            try:
                # Build request
                params = {"query": query}
                if package:
                    params["package"] = package
                
                # Make API request
                start_time = time.time()
                response = requests.get(
                    "http://127.0.0.1:8000/api/docs/search", 
                    params=params,
                    timeout=120
                )
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        # Display results
                        st.success(f"✅ Answer generated in {elapsed_time:.2f} seconds")
                        
                        # Display the answer
                        st.subheader("💡 Answer")
                        answer_text = result.get("result", "No answer available")
                        
                        # Split answer and sources if present
                        if "\n\nSources:" in answer_text:
                            answer_part, sources_part = answer_text.split("\n\nSources:", 1)
                            st.markdown(answer_part)
                            
                            # Display sources in an expander
                            with st.expander("📚 View Sources", expanded=True):
                                st.markdown(f"**Sources:** {sources_part}")
                        else:
                            st.markdown(answer_text)
                        
                        # Display metadata
                        with st.expander("🔍 Query Details"):
                            st.json({
                                "query": result.get("query"),
                                "package_filter": result.get("package"),
                                "response_time": f"{elapsed_time:.2f}s"
                            })
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.text(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request Timed Out**")
                st.info("The query is taking too long (>120s). This might happen if:\n- The OpenAI API is slow\n- The vector database is processing a large query\n- The backend is under heavy load\n\n💡 Try simplifying your question or try again in a moment.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 **Cannot Connect to API**")
                st.info("Make sure the backend is running:\n```bash\ndocker-compose up\n```\nOr check if the API is available at: http://127.0.0.1:8000")
            except requests.exceptions.RequestException as e:
                st.error(f"🌐 **Network Error**")
                st.info(f"Details: {str(e)}\n\nPlease check your network connection and ensure the API is accessible.")
            except json.JSONDecodeError:
                st.error("📄 **Invalid API Response**")
                st.info("The API returned an invalid response. This might indicate:\n- API is starting up\n- Configuration error\n- Unexpected server error\n\nCheck the API logs for more details.")
            except Exception as e:
                st.error(f"❌ **Unexpected Error**")
                st.info(f"An unexpected error occurred: {str(e)}\n\nIf this persists, please:\n1. Check API logs\n2. Verify OpenAI API key is set\n3. Ensure vector database is running")
                st.exception(e)  # Show full traceback in debug mode

# Example queries
st.markdown("---")
st.subheader("💡 Example Questions")

example_queries = [
    "What is BeautifulSoup and how do I use it?",
    "How do I create a DataFrame in pandas?",
    "What is NumPy used for?",
    "Explain Python decorators",
    "How do I read a CSV file with pandas?",
    "What are Python context managers?"
]

cols = st.columns(3)
for idx, example in enumerate(example_queries):
    with cols[idx % 3]:
        if st.button(example, key=f"example_{idx}"):
            st.rerun()
