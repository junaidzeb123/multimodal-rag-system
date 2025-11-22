import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import json

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
# Freeze parameters (NO TRAINING)
for p in projector.parameters():
    p.requires_grad = False


def embed_text(text: str):
    emb = text_model.encode(text, convert_to_tensor=True)
    emb = emb / emb.norm()  # L2 normalize
    return emb.cpu().numpy()


def embed_image(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)

    img_emb = img_emb.cpu().squeeze()

    # PROJECT to 768
    projected = projector(img_emb)
    projected = projected / projected.norm()  # normalize

    return projected.detach().numpy()


# FAISS cosine similarity index
index = faiss.IndexFlatIP(text_embedding_dim)

vectorstore = FAISS(
    embedding_function=None,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={},
    normalize_L2=True,
)


def add_document_to_faiss(doc: Document):
    # If text
    if doc.page_content and doc.page_content.strip() != "":
        emb = embed_text(doc.page_content)

    # If image
    elif "image_path" in doc.metadata:
        image = Image.open(doc.metadata["image_path"])
        emb = embed_image(image)

    else:
        raise ValueError("Document must have text or image_path")

    # Add to FAISS manually
    # Generate a unique ID for this document
    doc_id = str(len(vectorstore.index_to_docstore_id))

    # Add to FAISS index
    vectorstore.index.add(emb.reshape(1, -1))

    # Add to docstore
    vectorstore.docstore.add({doc_id: doc})

    # Map index to docstore ID
    vectorstore.index_to_docstore_id[len(vectorstore.index_to_docstore_id)] = doc_id


def index_chunks(chunks_file):
    """
    Index chunks into the global FAISS vectorstore

    Args:
        chunks_file: Path to chunks JSON file
    """
    with open(chunks_file, "r") as f:
        chunks = json.load(f)

    for chunk in chunks:
        if chunk["type"] == "text_chunk":
            doc = Document(
                page_content=chunk["content"],
                metadata={"type": "text", "source": chunks_file},
            )
            add_document_to_faiss(doc)

        elif chunk["type"] == "image_chunk":
            doc = Document(
                page_content="",  # text empty
                metadata={
                    "type": "image",
                    "source": chunks_file,
                    "ocr_text": chunk.get("ocr_text", ""),
                    "image_path": chunk["image_path"],
                },
            )
            add_document_to_faiss(doc)

    print(f"[OK] Indexed {len(chunks)} chunks from {chunks_file}")


def save_vectorstore(output_dir="vectordb", index_name="combined_faiss"):
    """
    Save the combined FAISS vectorstore to disk

    Args:
        output_dir: Directory to save FAISS index
        index_name: Name of the saved index
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, index_name)
    vectorstore.save_local(save_path)
    
    # Save the projector weights
    projector_path = os.path.join(output_dir, "projector.pt")
    torch.save(projector.state_dict(), projector_path)
    
    print(f"\n[OK] Combined FAISS vectorstore saved to {save_path}")
    print(f"[OK] Projector weights saved to {projector_path}")
    print(f"[OK] Total documents indexed: {len(vectorstore.index_to_docstore_id)}")

    return save_path
