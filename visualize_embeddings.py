"""
Embedding Visualization
=======================
Visualizes embedding space using t-SNE and UMAP.
Shows semantic clusters and proximity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
from pathlib import Path
from typing import List, Dict
from collections import Counter
from types import SimpleNamespace
import json

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    UMAP_AVAILABLE = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


def load_embeddings_and_metadata(vectorstore_path: str = "vectordb/combined_faiss/"):
    """Load embeddings alongside document metadata from a FAISS vector store."""
    vectorstore_path = Path(vectorstore_path)

    # Load FAISS index
    import faiss
    index_path = vectorstore_path / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(str(index_path))

    # Extract all embeddings
    n_vectors = index.ntotal
    embeddings = np.zeros((n_vectors, index.d), dtype=np.float32)
    for i in range(n_vectors):
        embeddings[i] = index.reconstruct(i)

    if n_vectors == 0:
        print("⚠️  Vector store is empty; no embeddings to visualize.")
        return embeddings, [], [], []

    # Attempt to load metadata
    docstore_path = vectorstore_path / "index.pkl"
    try:
        with open(docstore_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️  Metadata file missing at {docstore_path}; using placeholder labels.")
        data = {}
    except ModuleNotFoundError as exc:
        print(f"⚠️  {exc}. Metadata dependencies missing; using placeholder labels.")
        data = {}

    documents: List = []
    labels: List[str] = []
    doc_types: List[str] = []

    docstore = data.get('docstore') if isinstance(data, dict) else None
    store_dict = getattr(docstore, '_dict', {}) if docstore is not None else {}

    index_map = data.get('index_to_docstore_id') if isinstance(data, dict) else None
    ordered_ids: List = []

    if isinstance(index_map, list):
        ordered_ids = index_map
    elif isinstance(index_map, dict):
        if '_dict' in index_map:
            mapping = index_map['_dict']
            ordered_ids = [mapping.get(i) for i in range(len(mapping))]
        elif 'values' in index_map:
            ordered_ids = index_map['values']
        else:
            ordered_ids = [index_map.get(i) for i in range(n_vectors)]
    elif hasattr(index_map, '_dict'):
        mapping = index_map._dict
        ordered_ids = [mapping.get(i) for i in range(len(mapping))]

    if not ordered_ids and store_dict:
        ordered_ids = list(store_dict.keys())

    def resolve_doc(doc_id):
        if doc_id is None or not store_dict:
            return None
        if doc_id in store_dict:
            return store_dict[doc_id]
        if isinstance(doc_id, int):
            as_str = str(doc_id)
            if as_str in store_dict:
                return store_dict[as_str]
        if isinstance(doc_id, str) and doc_id.isdigit():
            as_int = int(doc_id)
            if as_int in store_dict:
                return store_dict[as_int]
        return None

    for idx in range(n_vectors):
        doc_id = ordered_ids[idx] if idx < len(ordered_ids) else None
        doc = resolve_doc(doc_id)
        if doc is None and store_dict:
            # Fall back to deterministic ordering if mapping is incomplete
            fallback_key = list(store_dict.keys())[idx % len(store_dict)]
            doc = resolve_doc(fallback_key) or store_dict.get(fallback_key)

        if doc is None:
            placeholder = SimpleNamespace(
                page_content="",
                metadata={'source': f'doc_{idx}', 'type': 'unknown'}
            )
            documents.append(placeholder)
            labels.append(f"doc_{idx}")
            doc_types.append("unknown")
        else:
            documents.append(doc)
            metadata = getattr(doc, 'metadata', {}) or {}
            source = metadata.get('source', f'doc_{idx}')
            labels.append(Path(source).stem.split('_chunk')[0])
            doc_types.append(metadata.get('type', 'text'))

    # Align metadata lengths with embeddings count
    expected = embeddings.shape[0]
    if len(labels) < expected:
        for idx in range(len(labels), expected):
            documents.append(SimpleNamespace(
                page_content="",
                metadata={'source': f'doc_{idx}', 'type': 'unknown'}
            ))
            labels.append(f"doc_{idx}")
            doc_types.append("unknown")
    elif len(labels) > expected:
        labels = labels[:expected]
        doc_types = doc_types[:expected]
        documents = documents[:expected]

    return embeddings, labels, doc_types, documents


def reduce_dimensions_tsne(embeddings: np.ndarray, n_components: int = 2, 
                          perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """
    Reduce embeddings to 2D using t-SNE
    
    Args:
        embeddings: High-dimensional embeddings (n_samples, n_features)
        n_components: Target dimensionality (2 or 3)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
    
    Returns:
        Reduced embeddings (n_samples, n_components)
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("At least two embeddings are required for t-SNE visualization.")

    effective_perplexity = min(perplexity, max(1, n_samples - 1))
    if effective_perplexity >= n_samples:
        effective_perplexity = max(1, n_samples - 1)

    print(f"Reducing {n_samples} embeddings to {n_components}D using t-SNE (perplexity={effective_perplexity})...")

    tsne = TSNE(n_components=n_components,
                perplexity=effective_perplexity,
                random_state=random_state,
                max_iter=1000,
                verbose=1)

    embeddings_2d = tsne.fit_transform(embeddings)

    print("✅ t-SNE reduction complete")
    return embeddings_2d


def reduce_dimensions_umap(embeddings: np.ndarray, n_components: int = 2, 
                          n_neighbors: int = 15, min_dist: float = 0.1, 
                          random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP if the dependency is available."""
    if not UMAP_AVAILABLE:
        raise RuntimeError("UMAP is not installed. Install umap-learn or run with --method tsne." )

    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("At least two embeddings are required for UMAP visualization.")

    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))

    print(f"Reducing {n_samples} embeddings to {n_components}D using UMAP (n_neighbors={effective_neighbors})...")

    reducer = UMAP(n_components=n_components, n_neighbors=effective_neighbors,
                   min_dist=min_dist, random_state=random_state, verbose=True)

    embeddings_2d = reducer.fit_transform(embeddings)

    print("✅ UMAP reduction complete")
    return embeddings_2d


def plot_embeddings_by_source(embeddings_2d: np.ndarray, labels: List[str], 
                              title: str = "Embedding Space Visualization",
                              method: str = "t-SNE",
                              save_path: str = None):
    """
    Plot 2D embeddings colored by source document
    """
    if embeddings_2d.size == 0:
        print("⚠️  No embeddings available to plot by source.")
        return

    if not labels:
        labels = [f"doc_{idx}" for idx in range(len(embeddings_2d))]

    plt.figure(figsize=(14, 10))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_labels))))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        if not mask.any():
            continue
        color = colors[i % len(colors)]
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color], label=label, alpha=0.6, s=50,
                    edgecolors='black', linewidth=0.5)
    
    plt.xlabel(f"{method} Component 1", fontsize=12)
    plt.ylabel(f"{method} Component 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title="Source Document", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_embeddings_by_type(embeddings_2d: np.ndarray, doc_types: List[str], 
                            title: str = "Embedding Space by Content Type",
                            method: str = "t-SNE",
                            save_path: str = None):
    """
    Plot 2D embeddings colored by document type (text, table, image)
    """
    if embeddings_2d.size == 0:
        print("⚠️  No embeddings available to plot by content type.")
        return

    if not doc_types:
        doc_types = ["unknown" for _ in range(len(embeddings_2d))]

    plt.figure(figsize=(12, 8))

    # Define colors for each type
    type_colors = {
        'text': '#3498db',
        'table': '#e74c3c',
        'image': '#2ecc71',
        'text_chunk': '#3498db',
        'table_chunk': '#e74c3c',
        'image_chunk': '#2ecc71'
    }
    
    unique_types = sorted(set(doc_types))

    for doc_type in unique_types:
        mask = np.array(doc_types) == doc_type
        if not mask.any():
            continue
        color = type_colors.get(doc_type, '#95a5a6')
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=color, label=doc_type, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    plt.xlabel(f"{method} Component 1", fontsize=12)
    plt.ylabel(f"{method} Component 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title="Content Type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_density_heatmap(embeddings_2d: np.ndarray, title: str = "Embedding Density Heatmap",
                        save_path: str = None):
    """
    Plot density heatmap of embeddings to show clustering patterns
    """
    if embeddings_2d.size == 0:
        print("⚠️  No embeddings available to create a density heatmap.")
        return

    plt.figure(figsize=(12, 8))
    
    # Create 2D histogram
    plt.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label='Number of Points')
    
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap saved to {save_path}")

    plt.show()
    plt.close()


def analyze_semantic_proximity(embeddings: np.ndarray, labels: List[str], 
                               documents: List, top_k: int = 10) -> Dict:
    """
    Analyze semantic proximity between documents
    Find most similar and most distant document pairs
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\nAnalyzing semantic proximity...")
    
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Find most similar pairs (excluding self-similarity)
    np.fill_diagonal(similarities, -1)  # Mask diagonal
    
    most_similar_pairs = []
    for i in range(len(embeddings)):
        j = np.argmax(similarities[i])
        if similarities[i, j] > 0:
            most_similar_pairs.append((i, j, similarities[i, j]))
    
    # Sort by similarity
    most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Find most distant pairs
    most_distant_pairs = []
    for i in range(len(embeddings)):
        j = np.argmin(similarities[i])
        most_distant_pairs.append((i, j, similarities[i, j]))
    
    most_distant_pairs.sort(key=lambda x: x[2])
    
    # Print analysis
    print(f"\n{'='*80}")
    print("SEMANTIC PROXIMITY ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Top {min(top_k, len(most_similar_pairs))} Most Similar Document Pairs:")
    print("-" * 80)
    for i, (idx1, idx2, sim) in enumerate(most_similar_pairs[:top_k], 1):
        doc1 = documents[idx1] if idx1 < len(documents) else SimpleNamespace(page_content="", metadata={})
        doc2 = documents[idx2] if idx2 < len(documents) else SimpleNamespace(page_content="", metadata={})
        doc1_preview = (getattr(doc1, 'page_content', '') or '')[:100]
        doc2_preview = (getattr(doc2, 'page_content', '') or '')[:100]
        print(f"{i}. Similarity: {sim:.4f}")
        print(f"   Doc1 [{labels[idx1]}]: {doc1_preview}...")
        print(f"   Doc2 [{labels[idx2]}]: {doc2_preview}...")
        print()
    
    print(f"\nTop {min(top_k, len(most_distant_pairs))} Most Distant Document Pairs:")
    print("-" * 80)
    for i, (idx1, idx2, sim) in enumerate(most_distant_pairs[:top_k], 1):
        doc1 = documents[idx1] if idx1 < len(documents) else SimpleNamespace(page_content="", metadata={})
        doc2 = documents[idx2] if idx2 < len(documents) else SimpleNamespace(page_content="", metadata={})
        doc1_preview = (getattr(doc1, 'page_content', '') or '')[:100]
        doc2_preview = (getattr(doc2, 'page_content', '') or '')[:100]
        print(f"{i}. Similarity: {sim:.4f}")
        print(f"   Doc1 [{labels[idx1]}]: {doc1_preview}...")
        print(f"   Doc2 [{labels[idx2]}]: {doc2_preview}...")
        print()
    
    return {
        'most_similar': most_similar_pairs[:top_k],
        'most_distant': most_distant_pairs[:top_k],
        'mean_similarity': np.mean(similarities[similarities > -1]),
        'std_similarity': np.std(similarities[similarities > -1])
    }


def save_visualization_report(proximity_analysis: Dict, output_path: str = "visualization_report.json"):
    """Save visualization analysis to JSON"""
    report = {
        'mean_similarity': float(proximity_analysis['mean_similarity']),
        'std_similarity': float(proximity_analysis['std_similarity']),
        'num_most_similar_pairs': len(proximity_analysis['most_similar']),
        'num_most_distant_pairs': len(proximity_analysis['most_distant'])
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Visualization report saved to {output_path}")


def plot_source_distribution(labels: List[str], output_path: Path):
    """Plot the number of embeddings contributed by each source document."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    counts = Counter(labels)
    if not counts:
        return

    sources = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(max(8, len(sources) * 0.6), 5))
    bars = plt.bar(sources, values, color="#4c72b0")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Embeddings")
    plt.title("Embeddings per Source Document")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                 ha="center", va="bottom")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Source distribution chart saved to {output_path}")


def visualize_all(vectorstore_path: str = "vectordb/combined_faiss/",
                 output_dir: str = "visualizations/",
                 method: str = "both"):
    """Generate dimensionality reduction plots and proximity analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading embeddings and metadata...")
    embeddings, labels, doc_types, documents = load_embeddings_and_metadata(vectorstore_path)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    if embeddings.size == 0:
        return

    unique_sources = len(set(labels)) if labels else 0
    unique_types = len(set(doc_types)) if doc_types else 0
    print(f"Detected {unique_sources} unique sources and {unique_types} content types")

    do_tsne = method in ("tsne", "both")
    do_umap = method in ("umap", "both")

    if do_tsne:
        embeddings_tsne = reduce_dimensions_tsne(embeddings)
        plot_embeddings_by_source(embeddings_tsne, labels,
                                  title="t-SNE: Document Embeddings by Source",
                                  method="t-SNE",
                                  save_path=str(output_dir / "tsne_by_source.png"))
        plot_embeddings_by_type(embeddings_tsne, doc_types,
                               title="t-SNE: Document Embeddings by Type",
                               method="t-SNE",
                               save_path=str(output_dir / "tsne_by_type.png"))
        plot_density_heatmap(embeddings_tsne,
                            title="t-SNE: Embedding Density Heatmap",
                            save_path=str(output_dir / "tsne_density.png"))

    if do_umap:
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP not available; skipped UMAP visualizations. Install umap-learn to enable.")
        else:
            embeddings_umap = reduce_dimensions_umap(embeddings)
            plot_embeddings_by_source(embeddings_umap, labels,
                                      title="UMAP: Document Embeddings by Source",
                                      method="UMAP",
                                      save_path=str(output_dir / "umap_by_source.png"))
            plot_embeddings_by_type(embeddings_umap, doc_types,
                                   title="UMAP: Document Embeddings by Type",
                                   method="UMAP",
                                   save_path=str(output_dir / "umap_by_type.png"))
            plot_density_heatmap(embeddings_umap,
                                title="UMAP: Embedding Density Heatmap",
                                save_path=str(output_dir / "umap_density.png"))

    proximity_analysis = analyze_semantic_proximity(embeddings, labels, documents)
    save_visualization_report(proximity_analysis, str(output_dir / "proximity_analysis.json"))
    plot_source_distribution(labels, output_dir / "source_distribution.png")

    print("\nVisualization complete")
    print(f"All visualizations saved to: {output_dir}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize embedding space")
    parser.add_argument("--vectorstore-path", type=str, default="vectordb/combined_faiss/",
                       help="Path to FAISS vectorstore")
    parser.add_argument("--output-dir", type=str, default="visualizations/",
                       help="Output directory for visualizations")
    parser.add_argument("--method", type=str, choices=['tsne', 'umap', 'both'], default='both',
                       help="Dimensionality reduction method")
    
    args = parser.parse_args()
    
    visualize_all(args.vectorstore_path, args.output_dir, method=args.method)
