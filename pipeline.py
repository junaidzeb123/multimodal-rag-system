"""
PDF to Semantic Chunks Pipeline
================================
Orchestrates the complete PDF processing workflow:
1. Parse PDF (extract text, tables, images)
2. Build semantic chunks with overlap
3. Save results
4. Create embeddings and store in a single combined FAISS index

"""

import os
import json
from parser import extract_pdf
from chunker import build_semantic_chunks
from embedder import index_chunks, save_vectorstore
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "parsed_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunking parameters
MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 200
OVERLAP_CHARS = 100


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def process_pdf_to_chunks(pdf_path):
    """
    Complete pipeline: PDF → Extraction → Semantic Chunks
    """
    print(f"\n{'=' * 70}")
    print(f"PROCESSING: {os.path.basename(pdf_path)}")
    print(f"{'=' * 70}")

    # Step 1: Extract PDF content
    raw_blocks, pdf_name = extract_pdf(pdf_path)

    # Step 2: Build semantic chunks
    print("\n[2/3] Building semantic chunks...")
    chunks = build_semantic_chunks(
        raw_blocks,
        max_chunk_chars=MAX_CHUNK_CHARS,
        min_chunk_chars=MIN_CHUNK_CHARS,
        overlap=OVERLAP_CHARS,
    )

    # Step 3: Save chunks
    print("[3/3] Saving chunks...")
    chunk_file = os.path.join(OUTPUT_DIR, f"{pdf_name}_chunks.json")
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"  ✓ Created {len(chunks)} semantic chunks")
    print(f"  ✓ Saved to: {chunk_file}")
    print(f"\n{'=' * 70}")
    print("✓ PIPELINE COMPLETE!")
    print(f"{'=' * 70}\n")

    return chunks, chunk_file


def create_embedding_and_store_in_faiss(file_path):
    """
    Create embeddings for chunks and store in FAISS vector store
    """

    print(f"\n[EMBEDDER] Creating embeddings and storing to FAISS for {file_path}...")
    index_chunks(file_path)


# ============================================================================
# CLI INTERFACE
# ============================================================================
if __name__ == "__main__":
    # Process all PDFs in Data directory
    data_dir = "./Data"
    pdf_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF files to process\n")

    for pdf_path in pdf_files:
        process_pdf_to_chunks(pdf_path)

    # Create embeddings for all chunks and save to vector store
    chunks_files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith("_chunks.json")
    ]

    print(f"\n{'=' * 70}")
    print("CREATING COMBINED FAISS INDEX")
    print(f"{'=' * 70}\n")

    # Index all chunks into a single FAISS vectorstore
    for chunk_file in chunks_files:
        create_embedding_and_store_in_faiss(chunk_file)

    # Save the combined FAISS index
    save_vectorstore(output_dir="vectordb", index_name="combined_faiss")

    print(f"\n{'=' * 70}")
    print("✓ ALL PROCESSING COMPLETE!")
    print(f"{'=' * 70}\n")
