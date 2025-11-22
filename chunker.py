"""
Semantic Chunker Module
========================
Builds semantic chunks with overlap from raw extracted blocks.
"""

import os
import json


# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 200
OVERLAP_CHARS = 100


# ============================================================================
# CHUNKING FUNCTION
# ============================================================================
def build_semantic_chunks(raw_blocks, max_chunk_chars=MAX_CHUNK_CHARS, 
                         min_chunk_chars=MIN_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """
    Build semantic chunks with overlap from raw blocks
    
    Args:
        raw_blocks: List of extracted blocks from parser
        max_chunk_chars: Maximum characters per text chunk
        min_chunk_chars: Minimum characters to create a chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        list: Semantic chunks ready for embedding
    """
    chunks = []
    current_text = ""

    def push_text_chunk():
        nonlocal current_text

        clean = current_text.strip()

        if len(clean) > min_chunk_chars:
            # Save chunk
            chunks.append({"type": "text_chunk", "content": clean})

            # Apply overlap: keep last N chars
            if len(clean) > overlap:
                current_text = clean[-overlap:]
            else:
                current_text = ""
        else:
            current_text = ""

    # Process raw blocks
    for item in raw_blocks:

        # TEXT BLOCK HANDLING
        if item["type"] == "text":
            text = item["content"]

            # Skip useless fragments
            if len(text) < 15:
                continue

            # Skip headers/footers (customize as needed)
            if any(
                h in text.lower()
                for h in ["pwc", "confidential", "table of contents", "copyright"]
            ):
                continue

            # Accumulate text until max chunk size reached
            if len(current_text) + len(text) < max_chunk_chars:
                current_text += " " + text
            else:
                push_text_chunk()
                current_text += " " + text

        # TABLE HANDLING
        elif item["type"] == "table":
            push_text_chunk()
            chunks.append({"type": "table_chunk", "content": item["content"]})

        # IMAGE HANDLING
        elif item["type"] == "image":
            ocr = item["ocr_text"].strip()

            if len(ocr) < 20:
                continue

            push_text_chunk()
            chunks.append(
                {
                    "type": "image_chunk",
                    "image_path": item["image_path"],
                    "ocr_text": ocr,
                }
            )

    # Push remaining accumulated text
    push_text_chunk()

    return chunks


def chunk_from_json(json_path, output_dir="parsed_output", **chunk_params):
    """
    Load parsed JSON and create semantic chunks
    
    Args:
        json_path: Path to parsed JSON file
        output_dir: Directory to save chunks
        **chunk_params: Optional parameters for chunking (max_chunk_chars, etc.)
        
    Returns:
        tuple: (chunks_list, output_path)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_name = os.path.basename(json_path).replace("_parsed.json", "")
    
    print(f"\n[CHUNKER] Building semantic chunks for {pdf_name}...")
    chunks = build_semantic_chunks(data, **chunk_params)
    
    # Save chunks
    chunk_file = os.path.join(output_dir, f"{pdf_name}_chunks.json")
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    
    print(f"  ✓ Created {len(chunks)} semantic chunks")
    print(f"  ✓ Saved to: {chunk_file}")
    
    return chunks, chunk_file


# ============================================================================
# CLI INTERFACE (for standalone usage)
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <parsed_json_path>")
        print("Example: python chunker.py parsed_output/document_parsed.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    chunk_from_json(json_path)
