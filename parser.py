"""
PDF Parser Module
=================
Extracts text, tables, and images from PDF files with metadata.
"""

import os
import io
import json
import hashlib
import fitz
import pdfplumber
from PIL import Image
import pytesseract
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "parsed_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# IMAGE FILTERING
# ============================================================================
def is_useless_image(image_path):
    """Filter out low-quality, blank, or decorative images"""
    image = Image.open(image_path).convert("L")  # grayscale

    # 1. Check resolution
    w, h = image.size
    if w < 300 or h < 300:
        return True

    # 2. Check variance (blank/gradient images have variance near 0–10)
    arr = np.array(image)
    if arr.var() < 50:   # tune threshold if needed
        return True

    # 3. File size check
    if os.path.getsize(image_path) < 20 * 1024:  # 20 KB
        return True

    # 4. OCR content check
    text = pytesseract.image_to_string(image)
    if len(text.strip()) == 0:
        return True

    return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def hash_str(s):
    """Generate MD5 hash of string"""
    return hashlib.md5(s.encode()).hexdigest()


def save_image(image_bytes, page_num, pdf_name, output_dir=OUTPUT_DIR):
    """Save extracted image to disk"""
    image = Image.open(io.BytesIO(image_bytes))
    img_name = f"{pdf_name}_page{page_num}_{hash_str(str(image_bytes))}.png"
    img_path = os.path.join(output_dir, img_name)
    image.save(img_path)
    return img_path


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================
def extract_pdf(pdf_path, output_dir=OUTPUT_DIR):
    """
    Extract text, tables, and images from PDF
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images and metadata
        
    Returns:
        tuple: (raw_blocks_list, pdf_name)
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    data = []

    print(f"\n[PARSER] Extracting content from: {pdf_name}")

    # TABLE extraction with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        plumber_pages = pdf.pages

    for page_num, page in enumerate(doc):
        print(f"  → Processing page {page_num + 1}/{len(doc)}", end="\r")

        # ********************
        # 1. TEXT EXTRACTION
        # ********************
        text_blocks = page.get_text("blocks")

        for i, block in enumerate(text_blocks):
            text = block[4].strip()
            if not text:
                continue

            data.append({
                "type": "text",
                "content": text,
                "page": page_num + 1,
                "pdf": pdf_name,
                "bbox": block[:4]
            })

        # ********************
        # 2. TABLE EXTRACTION
        # ********************
        try:
            tables = plumber_pages[page_num].extract_tables()
            for t_idx, table in enumerate(tables):
                if table:
                    table_text = "\n".join([", ".join([str(cell) if cell else "" for cell in row]) for row in table])

                    data.append({
                        "type": "table",
                        "content": table_text,
                        "page": page_num + 1,
                        "pdf": pdf_name,
                        "bbox": None
                    })
        except Exception:
            pass  # Skip table extraction errors

        # ********************
        # 3. IMAGE EXTRACTION
        # ********************
        img_list = page.get_images(full=True)

        for img in img_list:
            xref = img[0]
            base_img = doc.extract_image(xref)
            image_bytes = base_img["image"]

            img_path = save_image(image_bytes, page_num + 1, pdf_name, output_dir)
            
            # Filter useless images
            if is_useless_image(img_path):
                os.remove(img_path)
                continue
                
            # OCR from image
            ocr_text = pytesseract.image_to_string(Image.open(img_path))

            data.append({
                "type": "image",
                "image_path": img_path,
                "ocr_text": ocr_text,
                "page": page_num + 1,
                "pdf": pdf_name,
                "bbox": None
            })

    print(f"\n  ✓ Extracted {len(data)} blocks")

    # Save parsed data
    output_json = os.path.join(output_dir, f"{pdf_name}_parsed.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"  ✓ Saved to: {output_json}")
    return data, pdf_name


# ============================================================================
# CLI INTERFACE (for standalone usage)
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    extract_pdf(pdf_path)
