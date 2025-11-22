"""
Streamlit RAG Interface
=======================
ChatGPT-like interface for the RAG system.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retriever import load_vectorstore, search_text
from generator import generate_answer, generate_answer_with_streaming, format_context
import dotenv

# Load environment variables
dotenv.load_dotenv()


# Page configuration
st.set_page_config(
    page_title="University RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f7f7f8;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 24px;
        border: 2px solid #e0e0e0;
    }
    
    /* User message bubble */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0;
        margin-left: 20%;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant message bubble */
    .assistant-message {
        background: white;
        color: #2c3e50;
        padding: 16px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px 0;
        margin-right: 20%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #10a37f;
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Source box styling */
    .source-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(253, 203, 110, 0.3);
        font-size: 0.9em;
        color: #2c3e50;
    }
    
    /* Context box styling */
    .context-box {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.88em;
        border: 1px solid #e0e0e0;
        color: #4a5568;
    }
    
    /* Scrollbar styling */
    .context-box::-webkit-scrollbar {
        width: 8px;
    }
    
    .context-box::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .context-box::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .context-box::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Avatar icons */
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: 700;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #10a37f 0%, #0d8c6c 100%);
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    with st.spinner('🔄 Loading vectorstore...'):
        try:
            st.session_state.vectorstore = load_vectorstore()
            st.success('✅ Vectorstore loaded successfully!')
        except Exception as e:
            st.error(f'❌ Error loading vectorstore: {e}')
            st.stop()

if 'show_context' not in st.session_state:
    st.session_state.show_context = False

if 'num_results' not in st.session_state:
    st.session_state.num_results = 5


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Number of retrieved documents
    num_results = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More documents provide more context but may include irrelevant information"
    )
    st.session_state.num_results = num_results
    
    # Show context toggle
    show_context = st.checkbox(
        "Show retrieved context",
        value=st.session_state.show_context,
        help="Display the retrieved document chunks used to generate the answer"
    )
    st.session_state.show_context = show_context
    
    # Show sources toggle
    show_sources = st.checkbox(
        "Show sources",
        value=True,
        help="Display source documents with relevance scores"
    )
    
    st.divider()
    
    # API Key status
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ GROQ_API_KEY not set")
        st.info("Set your API key in .env file")
    
    st.divider()
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info section
    st.markdown("### 📚 Available Documents")
    st.markdown("""
    - Annual Report 2023-24
    - Financial Statements
    - FYP Handbook 2023
    """)
    
    st.divider()
    
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "What is the FYP report format?",
        "Who is Dr. Atif Tahir?",
        "What are the FYP evaluation criteria?",
        "Tell me about the ACM chapter",
    ]
    
    for query in example_queries:
        if st.button(f"📝 {query}", key=query, use_container_width=True):
            st.session_state.query_to_send = query


# Main chat interface
st.title("🎓 University RAG Assistant")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
    <h3 style='margin: 0; color: white;'>💬 Ask questions about university documents and get AI-powered answers!</h3>
    <p style='margin: 8px 0 0 0; opacity: 0.9;'>Powered by CLIP embeddings and Groq LLM</p>
</div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <span class="user-avatar">👤 You</span>
            </div>
            <div style='line-height: 1.6;'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <span class="assistant-avatar">🤖 Assistant</span>
            </div>
            <div style='line-height: 1.6;'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if "sources" in message and show_sources:
            with st.expander("📑 View Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📄 Source {i}:</strong> {source['file']}<br>
                        <div style='margin-top: 6px;'>
                            <span style='background: rgba(255,255,255,0.8); padding: 3px 8px; border-radius: 6px; margin-right: 8px;'>
                                📌 {source['type']}
                            </span>
                            <span style='background: rgba(255,255,255,0.8); padding: 3px 8px; border-radius: 6px;'>
                                ⭐ {source['score']:.3f}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show context if available and enabled
        if "context" in message and st.session_state.show_context:
            with st.expander("📄 View Retrieved Context", expanded=False):
                context_html = message['context'].replace('\n', '<br>')
                st.markdown(f"""
                <div class="context-box">
                    {context_html}
                </div>
                """, unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask a question about university documents...")

# Handle example query button clicks
if 'query_to_send' in st.session_state:
    query = st.session_state.query_to_send
    del st.session_state.query_to_send

if query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message immediately
    st.markdown(f"""
    <div class="user-message">
        <strong>👤 You:</strong><br>
        {query}
    </div>
    """, unsafe_allow_html=True)
    
    # Process query
    with st.spinner("🔍 Searching documents..."):
        try:
            # Retrieve documents
            results = search_text(query, st.session_state.vectorstore, k=st.session_state.num_results)
            
            # Generate answer
            with st.spinner("💭 Generating answer..."):
                response = generate_answer(query, results)
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "context": response["context"]
            })
            
            # Rerun to display the new message
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}"
            })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                background-clip: text;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.1em;
                font-weight: 700;
                margin-bottom: 8px;'>
        🎓 University RAG Assistant
    </div>
    <div style='color: #666; font-size: 0.9em;'>
        Powered by CLIP + Groq LLM | Built with ❤️ using Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
