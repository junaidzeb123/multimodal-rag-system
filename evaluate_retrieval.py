"""
Retrieval Quality Evaluation
=============================
Evaluates retrieval performance using Precision@K, Recall@K, and MAP.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any
from retriever import load_vectorstore, search_text
from pathlib import Path


def load_ground_truth(filepath: str = "ground_truth_queries.json") -> Dict:
    """Load ground truth query-document pairs"""
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def create_ground_truth_dataset():
    """
    Create ground truth dataset for evaluation.
    Each query has a list of relevant document IDs.
    """
    ground_truth = {
        "queries": [
            {
                "id": 1,
                "query": "What is the FYP report format?",
                "type": "text",
                "relevant_docs": [
                    "3. FYP-Handbook-2023_chunk_",  # Prefix for relevant chunks
                ],
                "relevant_keywords": ["format", "report", "fyp", "structure"]
            },
            {
                "id": 2,
                "query": "Who is Dr. Atif Tahir?",
                "type": "text",
                "relevant_docs": [
                    "1. Annual Report 2023-24_chunk_",
                ],
                "relevant_keywords": ["atif", "tahir", "dr", "faculty"]
            },
            {
                "id": 3,
                "query": "What are the FYP evaluation criteria?",
                "type": "text",
                "relevant_docs": [
                    "3. FYP-Handbook-2023_chunk_",
                ],
                "relevant_keywords": ["evaluation", "criteria", "fyp", "assessment"]
            },
            {
                "id": 4,
                "query": "Tell me about the ACM chapter",
                "type": "text",
                "relevant_docs": [
                    "1. Annual Report 2023-24_chunk_",
                ],
                "relevant_keywords": ["acm", "chapter", "student", "society"]
            },
            {
                "id": 5,
                "query": "What are the financial highlights?",
                "type": "text",
                "relevant_docs": [
                    "2. financials_chunk_",
                    "1. Annual Report 2023-24_chunk_",
                ],
                "relevant_keywords": ["financial", "revenue", "budget", "highlights"]
            },
            {
                "id": 6,
                "query": "What is the university vision?",
                "type": "text",
                "relevant_docs": [
                    "1. Annual Report 2023-24_chunk_",
                ],
                "relevant_keywords": ["vision", "mission", "goals", "objectives"]
            },
            {
                "id": 7,
                "query": "FYP proposal submission deadline",
                "type": "text",
                "relevant_docs": [
                    "3. FYP-Handbook-2023_chunk_",
                ],
                "relevant_keywords": ["deadline", "submission", "proposal", "date"]
            },
            {
                "id": 8,
                "query": "Faculty achievements and publications",
                "type": "text",
                "relevant_docs": [
                    "1. Annual Report 2023-24_chunk_",
                ],
                "relevant_keywords": ["faculty", "achievements", "publications", "research"]
            },
            {
                "id": 9,
                "query": "Student enrollment statistics",
                "type": "text",
                "relevant_docs": [
                    "1. Annual Report 2023-24_chunk_",
                ],
                "relevant_keywords": ["enrollment", "students", "statistics", "admission"]
            },
            {
                "id": 10,
                "query": "FYP supervisor allocation process",
                "type": "text",
                "relevant_docs": [
                    "3. FYP-Handbook-2023_chunk_",
                ],
                "relevant_keywords": ["supervisor", "allocation", "fyp", "process"]
            }
        ]
    }
    
    # Save to file
    with open("ground_truth_queries.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"✅ Created ground truth dataset with {len(ground_truth['queries'])} queries")
    return ground_truth


def is_relevant(retrieved_doc_id: str, relevant_doc_prefixes: List[str], 
                retrieved_content: str, relevant_keywords: List[str]) -> bool:
    """
    Check if retrieved document is relevant based on:
    1. Document ID prefix matching
    2. Keyword presence in content
    """
    # Check document ID prefix
    for prefix in relevant_doc_prefixes:
        if retrieved_doc_id.startswith(prefix):
            return True
    
    # Check keyword presence (at least 2 keywords should match)
    content_lower = retrieved_content.lower()
    keyword_matches = sum(1 for kw in relevant_keywords if kw.lower() in content_lower)
    
    if keyword_matches >= 2:
        return True
    
    return False


def _normalize_retrieved_items(retrieved: List[Any]) -> List[Dict[str, Any]]:
    """Convert raw retrieval results into dicts with metadata/content fields."""
    normalized = []
    for item in retrieved:
        score = None
        doc = item
        if isinstance(item, tuple) and len(item) >= 1:
            doc = item[0]
            score = item[1] if len(item) > 1 else None

        metadata = {}
        content = ""

        if hasattr(doc, "metadata"):
            metadata = doc.metadata or {}
        elif isinstance(doc, dict):
            metadata = doc.get("metadata", {}) or {}

        if hasattr(doc, "page_content"):
            content = doc.page_content or ""
        elif isinstance(doc, dict):
            content = doc.get("content") or doc.get("page_content", "")

        if not content:
            content = metadata.get("ocr_text", "")

        normalized.append({
            "metadata": metadata,
            "content": content or "",
            "score": score,
        })

    return normalized


def precision_at_k(retrieved: List[Dict], relevant_docs: List[str], 
                   relevant_keywords: List[str], k: int) -> float:
    """
    Precision@K = (# of relevant docs in top-K) / K
    """
    if k == 0 or len(retrieved) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_count = 0
    
    for doc in top_k:
        doc_id = doc.get('metadata', {}).get('source', '')
        content = doc.get('content', '')
        if is_relevant(doc_id, relevant_docs, content, relevant_keywords):
            relevant_count += 1
    
    return relevant_count / k


def recall_at_k(retrieved: List[Dict], relevant_docs: List[str], 
                relevant_keywords: List[str], k: int, 
                total_relevant: int = None) -> float:
    """
    Recall@K = (# of relevant docs in top-K) / (total # of relevant docs)
    """
    if total_relevant is None or total_relevant == 0:
        # Estimate total relevant docs
        total_relevant = len(relevant_docs) * 5  # Assume ~5 chunks per doc
    
    if len(retrieved) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_count = 0
    
    for doc in top_k:
        doc_id = doc.get('metadata', {}).get('source', '')
        content = doc.get('content', '')
        if is_relevant(doc_id, relevant_docs, content, relevant_keywords):
            relevant_count += 1
    
    return relevant_count / total_relevant


def average_precision(retrieved: List[Dict], relevant_docs: List[str], 
                     relevant_keywords: List[str]) -> float:
    """
    Average Precision = mean of precision values at positions where relevant docs are found
    """
    if len(retrieved) == 0:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, doc in enumerate(retrieved, 1):
        doc_id = doc.get('metadata', {}).get('source', '')
        content = doc.get('content', '')
        
        if is_relevant(doc_id, relevant_docs, content, relevant_keywords):
            relevant_count += 1
            precision_at_i = relevant_count / i
            precisions.append(precision_at_i)
    
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)


def mean_average_precision(results: List[Tuple[str, List[Dict], List[str], List[str]]]) -> float:
    """
    MAP = mean of Average Precision scores across all queries
    """
    if len(results) == 0:
        return 0.0
    
    aps = []
    for query, retrieved, relevant_docs, relevant_keywords in results:
        ap = average_precision(retrieved, relevant_docs, relevant_keywords)
        aps.append(ap)
    
    return np.mean(aps)


def evaluate_retrieval(vectorstore, ground_truth: Dict, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
    """
    Evaluate retrieval quality across multiple queries and K values
    """
    results = {
        'precision_at_k': {k: [] for k in k_values},
        'recall_at_k': {k: [] for k in k_values},
        'average_precision': [],
        'query_results': []
    }
    
    print("\nRetrieval Evaluation")
    print("--------------------")
    
    all_results_for_map = []
    
    for query_data in ground_truth['queries']:
        query = query_data['query']
        relevant_docs = query_data['relevant_docs']
        relevant_keywords = query_data['relevant_keywords']
        query_type = query_data['type']
        
        print(f"Query {query_data['id']}: {query}")
        print(f"Type: {query_type}")
        
        # Retrieve documents
        if query_type == "text":
            retrieved_raw = search_text(query, vectorstore, k=max(k_values))
        elif query_type == "image":
            # For image queries, would use search_image
            continue
        else:
            continue

        retrieved = _normalize_retrieved_items(retrieved_raw)
        
        # Calculate metrics for different K values
        query_metrics = {
            'query': query,
            'precision_at_k': {},
            'recall_at_k': {}
        }
        
        for k in k_values:
            p_at_k = precision_at_k(retrieved, relevant_docs, relevant_keywords, k)
            r_at_k = recall_at_k(retrieved, relevant_docs, relevant_keywords, k)
            
            results['precision_at_k'][k].append(p_at_k)
            results['recall_at_k'][k].append(r_at_k)
            
            query_metrics['precision_at_k'][k] = p_at_k
            query_metrics['recall_at_k'][k] = r_at_k
            
            print(f"  Precision@{k}: {p_at_k:.3f} | Recall@{k}: {r_at_k:.3f}")
        
        # Calculate Average Precision
        ap = average_precision(retrieved, relevant_docs, relevant_keywords)
        results['average_precision'].append(ap)
        query_metrics['average_precision'] = ap

        print(f"  Average Precision: {ap:.3f}\n")

        results['query_results'].append(query_metrics)
    all_results_for_map.append((query, retrieved, relevant_docs, relevant_keywords))
    
    # Calculate MAP
    map_score = mean_average_precision(all_results_for_map)
    results['map'] = map_score
    
    # Calculate mean metrics
    results['mean_precision_at_k'] = {k: np.mean(results['precision_at_k'][k]) 
                                       for k in k_values}
    results['mean_recall_at_k'] = {k: np.mean(results['recall_at_k'][k]) 
                                    for k in k_values}
    results['mean_average_precision'] = np.mean(results['average_precision'])
    
    return results


def print_evaluation_summary(results: Dict):
    """Print evaluation results in a formatted table"""
    print("\nEvaluation Summary")
    print("------------------")
    
    print("Mean Precision@K and Recall@K:")
    print(f"{'K':<10} {'Precision@K':<20} {'Recall@K':<20}")
    print("-" * 50)
    
    for k in sorted(results['mean_precision_at_k'].keys()):
        precision = results['mean_precision_at_k'][k]
        recall = results['mean_recall_at_k'][k]
        print(f"{k:<10} {precision:<20.3f} {recall:<20.3f}")
    
    print(f"\nMean Average Precision (MAP): {results['mean_average_precision']:.3f}\n")


def save_evaluation_results(results: Dict, filepath: str = "evaluation_results.json"):
    """Save evaluation results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {
        'mean_precision_at_k': {k: float(v) for k, v in results['mean_precision_at_k'].items()},
        'mean_recall_at_k': {k: float(v) for k, v in results['mean_recall_at_k'].items()},
        'mean_average_precision': float(results['mean_average_precision']),
        'map': float(results['map']),
        'query_results': results['query_results']
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"✅ Evaluation results saved to {filepath}")


def plot_mean_precision_recall(results: Dict, filepath: Path):
    """Generate grouped bar chart for mean Precision@K and Recall@K."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    k_values = sorted(results['mean_precision_at_k'].keys())
    precisions = [results['mean_precision_at_k'][k] for k in k_values]
    recalls = [results['mean_recall_at_k'][k] for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, precisions, width, label='Precision', color='#4c72b0')
    plt.bar(x + width / 2, recalls, width, label='Recall', color='#dd8452')

    plt.xlabel('K')
    plt.ylabel('Score')
    plt.title('Mean Precision@K and Recall@K')
    plt.xticks(x, [str(k) for k in k_values])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✅ Precision/Recall chart saved to {filepath}")


def plot_average_precision_per_query(results: Dict, filepath: Path):
    """Generate horizontal bar chart of Average Precision per query."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    queries = [entry['query'] for entry in results['query_results']]
    aps = [entry.get('average_precision', 0.0) for entry in results['query_results']]

    if not queries:
        return

    y_pos = np.arange(len(queries))

    plt.figure(figsize=(10, max(6, len(queries) * 0.5)))
    plt.barh(y_pos, aps, color='#55a868')
    plt.yticks(y_pos, queries)
    plt.xlabel('Average Precision')
    plt.title('Average Precision by Query')
    plt.xlim(0, 1)
    for idx, value in enumerate(aps):
        plt.text(value + 0.02, idx, f"{value:.2f}", va='center')
    plt.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✅ Average Precision chart saved to {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--vectorstore-path", type=str, default="vectordb/combined_faiss/",
                       help="Path to FAISS vectorstore")
    parser.add_argument("--create-gt", action="store_true",
                       help="Create ground truth dataset")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10],
                       help="K values for evaluation")
    parser.add_argument("--output-dir", type=str, default="retrieval_analysis",
                       help="Directory to store evaluation artifacts")
    
    args = parser.parse_args()
    
    # Create ground truth if requested
    if args.create_gt:
        ground_truth = create_ground_truth_dataset()
    else:
        ground_truth = load_ground_truth()
        if ground_truth is None:
            print("❌ Ground truth not found. Creating new dataset...")
            ground_truth = create_ground_truth_dataset()
    
    # Load vectorstore
    print(f"\n📚 Loading vectorstore from {args.vectorstore_path}...")
    vectorstore = load_vectorstore(args.vectorstore_path)
    
    # Evaluate retrieval
    results = evaluate_retrieval(vectorstore, ground_truth, k_values=args.k_values)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_evaluation_results(results, filepath=output_dir / "evaluation_results.json")
    plot_mean_precision_recall(results, output_dir / "precision_recall.png")
    plot_average_precision_per_query(results, output_dir / "average_precision_per_query.png")
