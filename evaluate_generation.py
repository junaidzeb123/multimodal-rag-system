"""
Generation Quality Evaluation
==============================
Evaluates LLM-generated answers using BLEU, ROUGE, cosine similarity,
and response latency metrics.
"""

import json
import numpy as np
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

# Import evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk', 'rouge-score'])
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model for semantic similarity
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def create_reference_answers():
    """
    Create reference answers for evaluation.
    These serve as ground truth for comparing generated answers.
    """
    reference_dataset = {
        "questions": [
            {
                "id": 1,
                "question": "What is the FYP report format?",
                "reference_answer": "The Final Year Project (FYP) report should follow a structured format including: title page, abstract, introduction, literature review, methodology, results, discussion, conclusion, and references. The report must be professionally formatted with proper citations and should typically be between 40-60 pages.",
                "key_points": ["title page", "abstract", "introduction", "methodology", "results", "conclusion", "references", "40-60 pages"]
            },
            {
                "id": 2,
                "question": "Who is Dr. Atif Tahir?",
                "reference_answer": "Dr. Atif Tahir is the Head of the Computer Science Department. He is a distinguished faculty member with extensive research experience in Artificial Intelligence and Machine Learning, having published over 50 research papers in international journals and conferences.",
                "key_points": ["Head", "Computer Science Department", "research", "AI", "Machine Learning", "publications"]
            },
            {
                "id": 3,
                "question": "What are the FYP evaluation criteria?",
                "reference_answer": "FYP evaluation criteria include: technical complexity (30%), innovation and originality (25%), implementation quality (20%), documentation and report quality (15%), and presentation skills (10%). The total marks are 100, and students must score at least 50% to pass.",
                "key_points": ["technical complexity", "innovation", "implementation", "documentation", "presentation", "50%", "pass"]
            },
            {
                "id": 4,
                "question": "Tell me about the ACM chapter",
                "reference_answer": "The ACM (Association for Computing Machinery) student chapter at the university is an active organization that organizes technical workshops, coding competitions, guest lectures, and networking events. In 2023-24, the chapter conducted 12 technical workshops covering various topics in computer science and technology.",
                "key_points": ["ACM", "student chapter", "workshops", "competitions", "lectures", "12 workshops", "2023-24"]
            },
            {
                "id": 5,
                "question": "What are the financial highlights?",
                "reference_answer": "The financial highlights include strong revenue growth, increased investment in research and infrastructure, scholarship allocations for deserving students, and improved financial sustainability. The university demonstrated fiscal responsibility with balanced budgets and strategic resource allocation.",
                "key_points": ["revenue", "growth", "investment", "research", "infrastructure", "scholarships", "budget"]
            }
        ]
    }
    
    # Save to file
    with open("reference_answers.json", 'w') as f:
        json.dump(reference_dataset, f, indent=2)
    
    print(f"✅ Created reference dataset with {len(reference_dataset['questions'])} Q&A pairs")
    return reference_dataset


def load_reference_answers(filepath: str = "reference_answers.json") -> Dict:
    """Load reference answers from file"""
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def calculate_bleu_score(reference: str, generated: str) -> Dict[str, float]:
    """
    Calculate BLEU scores (BLEU-1 through BLEU-4)
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    between generated and reference text.
    """
    # Tokenize
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    
    # Smoothing function to handle zero n-gram matches
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU scores for different n-grams
    bleu_scores = {
        'bleu-1': sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        'bleu-2': sentence_bleu([reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        'bleu-3': sentence_bleu([reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing),
        'bleu-4': sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
    }
    
    return bleu_scores


def calculate_rouge_scores(reference: str, generated: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    recall of n-grams and longest common subsequences.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    rouge_scores = {
        'rouge-1-f': scores['rouge1'].fmeasure,
        'rouge-1-p': scores['rouge1'].precision,
        'rouge-1-r': scores['rouge1'].recall,
        'rouge-2-f': scores['rouge2'].fmeasure,
        'rouge-2-p': scores['rouge2'].precision,
        'rouge-2-r': scores['rouge2'].recall,
        'rouge-l-f': scores['rougeL'].fmeasure,
        'rouge-l-p': scores['rougeL'].precision,
        'rouge-l-r': scores['rougeL'].recall,
    }
    
    return rouge_scores


def calculate_semantic_similarity(reference: str, generated: str) -> float:
    """
    Calculate cosine similarity between reference and generated answer embeddings.
    
    Measures semantic similarity regardless of exact wording.
    """
    # Generate embeddings
    ref_embedding = embedding_model.encode([reference])
    gen_embedding = embedding_model.encode([generated])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
    
    return float(similarity)


def calculate_key_point_coverage(reference_key_points: List[str], generated: str) -> float:
    """
    Calculate what percentage of key points are mentioned in the generated answer.
    """
    generated_lower = generated.lower()
    covered_points = sum(1 for point in reference_key_points 
                        if point.lower() in generated_lower)
    
    coverage = covered_points / len(reference_key_points) if reference_key_points else 0
    return coverage


def evaluate_single_answer(question: str, generated_answer: str, 
                          reference_answer: str, key_points: List[str],
                          latency: float = None) -> Dict:
    """
    Evaluate a single generated answer against reference.
    """
    # Calculate all metrics
    bleu_scores = calculate_bleu_score(reference_answer, generated_answer)
    rouge_scores = calculate_rouge_scores(reference_answer, generated_answer)
    semantic_sim = calculate_semantic_similarity(reference_answer, generated_answer)
    key_point_coverage = calculate_key_point_coverage(key_points, generated_answer)
    
    # Answer length metrics
    ref_length = len(reference_answer.split())
    gen_length = len(generated_answer.split())
    length_ratio = gen_length / ref_length if ref_length > 0 else 0
    
    results = {
        'question': question,
        'generated_answer': generated_answer,
        'reference_answer': reference_answer,
        'metrics': {
            **bleu_scores,
            **rouge_scores,
            'semantic_similarity': semantic_sim,
            'key_point_coverage': key_point_coverage,
            'reference_length': ref_length,
            'generated_length': gen_length,
            'length_ratio': length_ratio,
        }
    }
    
    if latency is not None:
        results['metrics']['latency'] = latency
    
    return results


def evaluate_generation_quality(reference_dataset: Dict, generated_answers: Dict,
                                latencies: Dict = None) -> Dict:
    """
    Evaluate generation quality across multiple Q&A pairs.
    
    Args:
        reference_dataset: Reference questions and answers
        generated_answers: Dict mapping question IDs to generated answers
        latencies: Dict mapping question IDs to response latencies
    
    Returns:
        Evaluation results with individual and aggregate metrics
    """
    results = {
        'individual_results': [],
        'aggregate_metrics': defaultdict(list)
    }
    
    print(f"\n{'='*80}")
    print("GENERATION QUALITY EVALUATION")
    print(f"{'='*80}\n")
    
    for qa_pair in reference_dataset['questions']:
        q_id = qa_pair['id']
        question = qa_pair['question']
        reference = qa_pair['reference_answer']
        key_points = qa_pair['key_points']
        
        if q_id not in generated_answers:
            print(f"⚠️  Skipping Q{q_id}: No generated answer found")
            continue
        
        generated = generated_answers[q_id]
        latency = latencies.get(q_id) if latencies else None
        
        print(f"Q{q_id}: {question}")
        
        # Evaluate
        eval_result = evaluate_single_answer(question, generated, reference, 
                                            key_points, latency)
        
        results['individual_results'].append(eval_result)
        
        # Aggregate metrics
        for metric, value in eval_result['metrics'].items():
            results['aggregate_metrics'][metric].append(value)
        
        # Print key metrics
        metrics = eval_result['metrics']
        print(f"  BLEU-4: {metrics['bleu-4']:.3f} | ROUGE-L: {metrics['rouge-l-f']:.3f} | " 
              f"Semantic Sim: {metrics['semantic_similarity']:.3f} | "
              f"Key Points: {metrics['key_point_coverage']:.1%}")
        if latency:
            print(f"  Latency: {latency:.3f}s")
        print()
    
    # Calculate mean metrics
    results['mean_metrics'] = {
        metric: np.mean(values) for metric, values in results['aggregate_metrics'].items()
    }
    
    return results


def print_evaluation_summary(results: Dict):
    """Print evaluation summary"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    metrics = results['mean_metrics']
    
    print("BLEU Scores:")
    print(f"  BLEU-1: {metrics['bleu-1']:.4f}")
    print(f"  BLEU-2: {metrics['bleu-2']:.4f}")
    print(f"  BLEU-3: {metrics['bleu-3']:.4f}")
    print(f"  BLEU-4: {metrics['bleu-4']:.4f}")
    
    print("\nROUGE Scores (F1):")
    print(f"  ROUGE-1: {metrics['rouge-1-f']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge-2-f']:.4f}")
    print(f"  ROUGE-L: {metrics['rouge-l-f']:.4f}")
    
    print("\nSemantic Metrics:")
    print(f"  Cosine Similarity: {metrics['semantic_similarity']:.4f}")
    print(f"  Key Point Coverage: {metrics['key_point_coverage']:.1%}")
    
    print("\nLength Metrics:")
    print(f"  Avg Reference Length: {metrics['reference_length']:.1f} words")
    print(f"  Avg Generated Length: {metrics['generated_length']:.1f} words")
    print(f"  Length Ratio: {metrics['length_ratio']:.2f}")
    
    if 'latency' in metrics:
        print("\nPerformance:")
        print(f"  Avg Response Latency: {metrics['latency']:.3f}s")
    
    print(f"\n{'='*80}\n")


def save_evaluation_results(results: Dict, filepath: str = "generation_evaluation.json"):
    """Save evaluation results to JSON"""
    # Simplify for JSON serialization
    results_to_save = {
        'mean_metrics': {k: float(v) for k, v in results['mean_metrics'].items()},
        'num_evaluated': len(results['individual_results']),
        'individual_results': [
            {
                'question': r['question'],
                'metrics': {k: float(v) for k, v in r['metrics'].items()}
            }
            for r in results['individual_results']
        ]
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✅ Evaluation results saved to {filepath}")


def plot_mean_scores(mean_metrics: Dict, filepath: Path):
    """Plot aggregate BLEU/ROUGE metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bleu_keys = ["bleu-1", "bleu-2", "bleu-3", "bleu-4"]
    rouge_keys = ["rouge-1-f", "rouge-2-f", "rouge-l-f"]

    scores = [mean_metrics.get(k, 0.0) for k in bleu_keys + rouge_keys]
    labels = [k.upper() for k in bleu_keys] + [k.replace("-f", "").upper() for k in rouge_keys]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color="#4c72b0")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Mean BLEU and ROUGE Scores")
    for idx, value in enumerate(scores):
        plt.text(idx, value + 0.01, f"{value:.2f}", ha="center")
    plt.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✅ Mean BLEU/ROUGE chart saved to {filepath}")


def plot_semantic_metrics(mean_metrics: Dict, filepath: Path):
    """Plot semantic similarity and key point coverage."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = [
        ("Semantic Similarity", mean_metrics.get("semantic_similarity", 0.0)),
        ("Key Point Coverage", mean_metrics.get("key_point_coverage", 0.0)),
    ]

    plt.figure(figsize=(6, 4))
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    plt.bar(labels, values, color="#55a868")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Semantic Alignment Metrics")
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.02, f"{value:.2f}", ha="center")
    plt.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✅ Semantic metrics chart saved to {filepath}")


def plot_latency_distribution(results: Dict, filepath: Path):
    """Plot latency distribution if available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    latencies = [r['metrics']['latency'] for r in results['individual_results'] if 'latency' in r['metrics']]
    if not latencies:
        return

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(latencies) + 1), latencies, color="#dd8452")
    plt.xlabel("Question #")
    plt.ylabel("Latency (s)")
    plt.title("Response Latency per Question")
    for idx, value in enumerate(latencies, start=1):
        plt.text(idx, value + 0.05, f"{value:.2f}", ha="center")
    plt.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✅ Latency chart saved to {filepath}")


def plot_per_question_metric(results: Dict, metric_key: str, title: str, ylabel: str, filepath: Path):
    """Plot per-question metric values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    questions = [res['question'] for res in results['individual_results']]
    values = [res['metrics'].get(metric_key, 0.0) for res in results['individual_results']]
    if not questions:
        return

    plt.figure(figsize=(10, max(5, len(questions) * 0.5)))
    y_pos = np.arange(len(questions))
    plt.barh(y_pos, values, color="#8172b2")
    plt.yticks(y_pos, questions)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.xlim(0, max(1.0, max(values) * 1.1))
    for idx, value in enumerate(values):
        plt.text(value + 0.02, idx, f"{value:.2f}", va='center')
    plt.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✅ {title} chart saved to {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate generation quality")
    parser.add_argument("--create-ref", action="store_true",
                       help="Create reference answers dataset")
    parser.add_argument("--generated-file", type=str,
                       help="JSON file with generated answers")
    parser.add_argument("--output-dir", type=str, default="generation_analysis",
                        help="Directory to store evaluation artifacts")
    
    args = parser.parse_args()
    
    # Create or load reference answers
    if args.create_ref or not Path("reference_answers.json").exists():
        reference_dataset = create_reference_answers()
    else:
        reference_dataset = load_reference_answers()
    
    # Example: Load generated answers from file
    if args.generated_file and Path(args.generated_file).exists():
        with open(args.generated_file, 'r') as f:
            generated_data = json.load(f)
            generated_answers = generated_data.get('answers', {})
            latencies = generated_data.get('latencies', {})
    else:
        # Example generated answers for demonstration
        print("\nNo generated answers file provided. Using example answers for demo...")
        generated_answers = {
            1: "The FYP report format includes a title page, abstract, introduction, literature review, methodology, results, discussion, and conclusion with references. Reports should be 40-60 pages.",
            2: "Dr. Atif Tahir heads the Computer Science Department and has extensive AI and ML research experience with many publications.",
            3: "FYP evaluation includes technical complexity (30%), innovation (25%), implementation (20%), documentation (15%), and presentation (10%). Students need 50% to pass.",
            4: "The ACM chapter organizes workshops, competitions, and lectures. They conducted 12 workshops in 2023-24.",
            5: "Financial highlights show revenue growth, research investment, infrastructure development, and scholarship programs with balanced budgets."
        }
        latencies = {1: 2.5, 2: 1.8, 3: 3.2, 4: 2.1, 5: 2.7}
    
    # Evaluate
    results = evaluate_generation_quality(reference_dataset, generated_answers, latencies)
    
    # Print and save results
    print_evaluation_summary(results)
    output_dir = Path(args.output_dir)
    save_evaluation_results(results, filepath=output_dir / "generation_evaluation.json")
    plot_mean_scores(results['mean_metrics'], output_dir / "mean_bleu_rouge.png")
    plot_semantic_metrics(results['mean_metrics'], output_dir / "semantic_metrics.png")
    plot_latency_distribution(results, output_dir / "latency_per_question.png")
    plot_per_question_metric(results, "bleu-4", "BLEU-4 by Question", "BLEU-4 Score",
                             output_dir / "bleu4_per_question.png")
    plot_per_question_metric(results, "rouge-l-f", "ROUGE-L by Question", "ROUGE-L F1",
                             output_dir / "rougeL_per_question.png")
    plot_per_question_metric(results, "semantic_similarity", "Semantic Similarity by Question",
                             "Cosine Similarity", output_dir / "semantic_similarity_per_question.png")
    plot_per_question_metric(results, "key_point_coverage", "Key Point Coverage by Question",
                             "Coverage", output_dir / "key_point_coverage_per_question.png")
