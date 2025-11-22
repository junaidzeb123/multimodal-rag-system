stre# Multimodal RAG System

A complete Retrieval-Augmented Generation (RAG) system for university documents with multimodal support (text and images).

## 🌟 Features

- **Multimodal Embeddings**: Uses CLIP for unified text and image embeddings (512-dim)
- **Image Search**: Upload images to find similar visual content in documents
- **Smart PDF Processing**: Extracts text, tables, and images with OCR
- **Semantic Chunking**: Intelligent text chunking with overlap for better context
- **Vector Search**: FAISS-based similarity search
- **LLM Generation**: Groq-powered answer generation
- **ChatGPT-like UI**: Beautiful Streamlit interface with text and image search modes

## 📁 Project Structure

```
multimodal-rag-system/
├── parser.py              # PDF extraction (text, tables, images)
├── chunker.py             # Semantic chunking with overlap
├── embedder_clip.py       # CLIP-based embeddings
├── retriever_clip.py      # Vector search and retrieval
├── generator.py           # LLM-based answer generation
├── pipeline_clip.py       # Complete indexing pipeline
├── rag_pipeline.py        # End-to-end RAG pipeline
├── streamlit_app.py       # Web interface
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (API keys)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd multimodal-rag-system

# Install dependencies
pip install -r requirements.txt

# Install system dependencies for OCR
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
# or
brew install tesseract  # macOS
```

### 2. Setup Environment

Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com

### 3. Prepare Documents

Place your PDF files in the `Data/` directory:
```
Data/
├── 1. Annual Report 2023-24.pdf
├── 2. financials.pdf
└── 3. FYP-Handbook-2023.pdf
```

### 4. Build Index

```bash
# Run the complete pipeline to create embeddings
python pipeline_clip.py
```

This will:
1. Extract content from PDFs (text, tables, images)
2. Create semantic chunks
3. Generate CLIP embeddings
4. Store in FAISS vector database

### 5. Run the Application

#### Option A: Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

#### Option B: Command Line

```bash
# Single question
python rag_pipeline.py "What is the FYP report format?"

# Interactive mode
python rag_pipeline.py --interactive

# Streaming mode
python rag_pipeline.py --stream "Your question here"

# Context only (no generation)
python rag_pipeline.py --context-only "Your query here"
```

## 💡 Usage Examples

### Web Interface

1. Open the app: `streamlit run streamlit_app.py`
2. **Text Search Mode** (default):
   - Type your question in the chat input
   - View AI-generated answers with sources
   - Try example queries like "What is the FYP report format?"
3. **Image Search Mode**:
   - Switch to "🖼️ Image Search" in the sidebar
   - Upload an image (PNG, JPG, JPEG, GIF, BMP)
   - Click "🔍 Search with this image" to find similar visual content
   - View retrieved documents with similar diagrams, charts, or images
4. Adjust settings in the sidebar:
   - Number of documents to retrieve (1-10)
   - Toggle "Show retrieved context"
   - Toggle "Show sources"

### Command Line

```bash
# Ask about FYP guidelines
python rag_pipeline.py "What are the FYP submission guidelines?"

# Query financial data
python rag_pipeline.py "What is the annual revenue for 2023?"

# Interactive session
python rag_pipeline.py -i
```

## 🔧 Configuration

### Chunking Parameters (in `pipeline_clip.py`)

```python
MAX_CHUNK_CHARS = 800    # Maximum characters per chunk
MIN_CHUNK_CHARS = 200    # Minimum characters to create a chunk
OVERLAP_CHARS = 100      # Overlap between chunks
```

### Retrieval Settings (in Streamlit sidebar)

- **Search Mode**: Text Search or Image Search
- **Number of documents**: 1-10 (default: 5)
- **Show context**: Toggle retrieved document chunks
- **Show sources**: Toggle source attribution
- **Image Upload** (Image Search mode): Upload images to find similar content

### LLM Settings (in `generator.py`)

```python
model="openai/gpt-oss-120b"  # Groq model
temperature=0.2               # Lower = more focused
max_tokens=2048              # Maximum response length
```

## 📊 System Architecture

```
┌─────────────┐
│   PDFs      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  PDF Parser                 │
│  - Extract text/tables      │
│  - Extract images (OCR)     │
│  - Filter low-quality imgs  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Semantic Chunker           │
│  - Chunk text (800 chars)   │
│  - Overlap (100 chars)      │
│  - Preserve tables/images   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  CLIP Embedder              │
│  - Text → 512-dim vector    │
│  - Image → 512-dim vector   │
│  - Handle 77 token limit    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  FAISS Vector Store         │
│  - Store embeddings         │
│  - Fast similarity search   │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    │   Query     │
    └──────┬──────┘
           │
           ▼
┌─────────────────────────────┐
│  Retriever                  │
│  - Embed query with CLIP    │
│  - Search FAISS index       │
│  - Return top-k docs        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Generator (LLM)            │
│  - Format context           │
│  - Generate answer          │
│  - Add source attribution   │
└──────────┬──────────────────┘
           │
           ▼
    ┌──────────┐
    │  Answer  │
    └──────────┘
```

## 🎯 Key Components

### 1. Multimodal Search

**Text Search:**
- Type natural language questions
- CLIP encodes text queries into 512-dim embeddings
- Searches across all document types (text, tables, images)
- Example: "What are the FYP evaluation criteria?"

**Image Search:**
- Upload any image (charts, diagrams, photos)
- CLIP encodes image into same 512-dim space as text
- Finds visually similar content in documents
- Use cases:
  - Find similar diagrams or flowcharts
  - Locate related charts or graphs
  - Search by organizational logos or photos
  - Identify similar document layouts

### 2. CLIP Embeddings
- **Why CLIP?** Unified embedding space for text and images
- **Dimension**: 512 (efficient and effective)
- **Token Limit**: Handles 77-token limit by chunking and averaging
- **Multimodal**: Text and images embedded in the same space

### 3. FAISS Vector Store
- **Index Type**: IndexFlatIP (inner product for cosine similarity)
- **Advantages**: Fast similarity search, memory efficient
- **Normalization**: L2-normalized embeddings

### 3. Semantic Chunking
- **Strategy**: Balance between context and specificity
- **Overlap**: Prevents information loss at boundaries
- **Multimodal**: Preserves tables and images as separate chunks

### 4. RAG Pipeline
- **Retrieval**: Top-k most relevant documents
- **Context**: Formatted with source attribution
- **Generation**: LLM creates answer from context only

## 🛠️ Troubleshooting

### FAISS Import Error
```bash
pip uninstall -y faiss-cpu faiss-gpu
pip install faiss-cpu
```

### OCR Issues
```bash
# Install tesseract
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

### Memory Issues
- Reduce `MAX_CHUNK_CHARS` in chunking
- Process PDFs one at a time
- Use smaller batch sizes for embeddings

### API Key Error
- Ensure `.env` file exists in the project root
- Check `GROQ_API_KEY` is set correctly
- Verify API key is valid at https://console.groq.com

## 📝 License

MIT License - feel free to use for academic or commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

Built with ❤️ using CLIP, FAISS, LangChain, and Streamlit
