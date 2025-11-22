"""
Retriever Module
================
Query the FAISS vectorstore and retrieve relevant documents.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from langchain_community.vectorstores import FAISS


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- TEXT MODEL (768-dim)
text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
text_embedding_dim = 768

# ---- IMAGE MODEL (512-dim)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_embedding_dim = 512

# ---- RANDOM PROJECTION: 512 → 768
projector = nn.Linear(image_embedding_dim, text_embedding_dim, bias=False)
projector.load_state_dict(torch.load("vectordb/projector.pt"))
# Freeze parameters (NO TRAINING)
for p in projector.parameters():
    p.requires_grad = False


def embed_text_query(text: str):
    """Embed text query for retrieval"""
    emb = text_model.encode(text, convert_to_tensor=True)
    emb = emb / emb.norm()  # L2 normalize
    return emb.cpu().numpy()


def embed_image_query(image: Image.Image):
    """Embed image query for retrieval"""
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)

    img_emb = img_emb.cpu().squeeze()

    # PROJECT to 768
    projected = projector(img_emb)
    projected = projected / projected.norm()  # normalize

    return projected.detach().numpy()


def load_vectorstore(vectorstore_path="vectordb/combined_faiss"):
    """
    Load the FAISS vectorstore from disk

    Args:
        vectorstore_path: Path to the saved FAISS vectorstore

    Returns:
        FAISS vectorstore object
    """
    print(f"[RETRIEVER] Loading vectorstore from {vectorstore_path}...")

    # Load with allow_dangerous_deserialization for local files
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings=None, allow_dangerous_deserialization=True
    )

    print(
        f"[OK] Loaded vectorstore with {len(vectorstore.index_to_docstore_id)} documents"
    )
    return vectorstore


def search_text(query: str, vectorstore, k=5):
    """
    Search for text query in the vectorstore

    Args:
        query: Text query string
        vectorstore: FAISS vectorstore object
        k: Number of results to return

    Returns:
        List of (Document, score) tuples
    """
    # Embed the query
    query_embedding = embed_text_query(query)

    # Perform similarity search
    scores, indices = vectorstore.index.search(query_embedding.reshape(1, -1), k)

    # Get documents
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:  # FAISS returns -1 for missing results
            continue
        doc_id = vectorstore.index_to_docstore_id[idx]
        doc = vectorstore.docstore.search(doc_id)
        results.append((doc, float(score)))

    return results


def search_image(image_path: str, vectorstore, k=5):
    """
    Search for similar content using an image query

    Args:
        image_path: Path to image file
        vectorstore: FAISS vectorstore object
        k: Number of results to return

    Returns:
        List of (Document, score) tuples
    """
    # Load and embed the image
    image = Image.open(image_path)
    query_embedding = embed_image_query(image)

    # Search in FAISS
    import numpy as np

    scores, indices = vectorstore.index.search(query_embedding.reshape(1, -1), k)

    # Get documents
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        doc_id = vectorstore.index_to_docstore_id[idx]
        doc = vectorstore.docstore.search(doc_id)
        results.append((doc, float(score)))

    return results


def print_results(results):
    """
    Pretty print search results

    Args:
        results: List of (Document, score) tuples
    """
    print(f"\n{'=' * 70}")
    print(f"SEARCH RESULTS ({len(results)} documents)")
    print(f"{'=' * 70}\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Result {i} (Score: {score:.4f}) ---")
        print(f"Type: {doc.metadata.get('type', 'unknown')}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")

        if doc.page_content:
            print(f"Content: {doc.page_content}...")
        elif "ocr_text" in doc.metadata:
            print(f"OCR Text: {doc.metadata['ocr_text']}...")

        if "image_path" in doc.metadata:
            print(f"Image: {doc.metadata['image_path']}")

        print()


# ============================================================================
# CLI INTERFACE
# ============================================================================
if __name__ == "__main__":
    import sys

    # Load vectorstore
    vectorstore = load_vectorstore()

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python retriever.py 'your query text'")
        print("  python retriever.py --image path/to/image.png")
        print("\nExample:")
        print("  python retriever.py 'What is the FYP report format?'")
        sys.exit(1)

    # Check if it's an image query
    if sys.argv[1] == "--image":
        if len(sys.argv) < 3:
            print("Error: Please provide image path")
            sys.exit(1)

        image_path = sys.argv[2]
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

        print(f"\n[QUERY] Image: {image_path}")
        results = search_image(image_path, vectorstore, k=k)
        print_results(results)

    else:
        # Text query
        query = " ".join(sys.argv[1:])
        k = 5  # Default number of results

        print(f"\n[QUERY] {query}")
        results = search_text(query, vectorstore, k=k)
        print_results(results)
