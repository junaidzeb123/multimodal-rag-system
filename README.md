# 🧠 Multimodal RAG System

A **Retrieval-Augmented Generation (RAG)** system that enables semantic search and Q&A over PDF documents using **both text and images**. Powered by CLIP embeddings, FAISS, and Groq.

## 🌟 Key Features
- **Multimodal Search:** Query documents using text *or* upload images to find similar visual content.
- **Unified Embeddings:** Uses **CLIP** to project text and images into a shared 512-dim vector space.
- **Smart Parsing:** Extracts text, tables, and images from PDFs using OCR (Tesseract).
- **High Performance:** **FAISS** for fast vector retrieval and **Groq** for instant LLM responses.
- **Interactive UI:** ChatGPT-style interface built with **Streamlit**.

## 🛠️ Tech Stack
- **Embeddings:** CLIP (OpenAI)
- **Vector Store:** FAISS
- **LLM:** Groq (Llama-3/Mixtral)
- **Frontend:** Streamlit
- **OCR:** Tesseract

## 🚀 Quick Start

### 1. Setup
```bash
# Clone & Install
git clone [https://github.com/yourusername/multimodal-rag-system.git](https://github.com/yourusername/multimodal-rag-system.git)
cd multimodal-rag-system
pip install -r requirements.txt

# Install Tesseract (Required for OCR)
sudo apt-get install tesseract-ocr  # Ubuntu
brew install tesseract              # macOS
