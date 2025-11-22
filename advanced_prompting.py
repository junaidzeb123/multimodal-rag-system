"""
Advanced Prompting Techniques
==============================
Implements Chain-of-Thought, Zero-shot, and Few-shot prompting for RAG.
Analyzes impact of different prompting strategies on answer quality.
"""

import os
import re
import time
import csv
from pathlib import Path
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import dotenv
import json

dotenv.load_dotenv()


def get_llm(model_name="openai/gpt-oss-120b", temperature=0.2):
    """Initialize LLM with specified parameters"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
        api_key=api_key,
    )


def format_context(results: List[Tuple[Document, float]]) -> str:
    """Format retrieved documents into context string"""
    context_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        source_file = source.split("/")[-1].replace("_chunks.json", "")
        content = doc.page_content if doc.page_content else doc.metadata.get("ocr_text", "")
        context_entry = f"[Source {i}: {source_file}]\n{content}\n"
        context_parts.append(context_entry)
    
    return "\n".join(context_parts)


# ============================================================================
# ZERO-SHOT PROMPTING
# ============================================================================

ZERO_SHOT_PROMPT = """You are a helpful AI assistant answering questions about university documents.

Context Information:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be concise and specific
- If information is insufficient, say so clearly

Answer:"""


def zero_shot_prompting(question: str, context_docs: List[Tuple[Document, float]], 
                        llm=None) -> Dict:
    """
    Zero-shot prompting: Direct question answering without examples.
    The model relies solely on its pre-trained knowledge and the given context.
    """
    if llm is None:
        llm = get_llm()
    
    context = format_context(context_docs)
    prompt = ZERO_SHOT_PROMPT.format(context=context, question=question)
    
    start_time = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    
    return {
        "answer": response.content,
        "prompt_type": "zero-shot",
        "latency": latency,
        "prompt": prompt
    }


# ============================================================================
# FEW-SHOT PROMPTING
# ============================================================================

FEW_SHOT_PROMPT = """You are a helpful AI assistant answering questions about university documents.

Here are some examples of how to answer questions based on context:

Example 1:
Context: The Final Year Project (FYP) must be submitted by April 30th, 2024. Late submissions will incur a penalty of 5% per day.
Question: What is the FYP submission deadline?
Answer: According to the FYP Handbook, the submission deadline is April 30th, 2024. Late submissions will be penalized at 5% per day.

Example 2:
Context: Dr. Atif Tahir is the Head of Computer Science Department. He has published over 50 research papers in AI and Machine Learning.
Question: Who is Dr. Atif Tahir?
Answer: Dr. Atif Tahir serves as the Head of the Computer Science Department and is a prolific researcher with over 50 publications in AI and Machine Learning, as mentioned in the Annual Report.

Example 3:
Context: The ACM student chapter organized 12 technical workshops in 2023-24, covering topics like Web Development, Mobile Apps, and Cloud Computing.
Question: What activities did the ACM chapter conduct?
Answer: The ACM chapter was highly active in 2023-24, organizing 12 technical workshops on various topics including Web Development, Mobile Apps, and Cloud Computing, according to the Annual Report.

Now answer this question:

Context Information:
{context}

Question: {question}

Answer (following the example format):"""


def few_shot_prompting(question: str, context_docs: List[Tuple[Document, float]], 
                       llm=None) -> Dict:
    """
    Few-shot prompting: Provides examples before the actual question.
    Helps the model understand the expected answer format and style.
    """
    if llm is None:
        llm = get_llm()
    
    context = format_context(context_docs)
    prompt = FEW_SHOT_PROMPT.format(context=context, question=question)
    
    start_time = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    
    return {
        "answer": response.content,
        "prompt_type": "few-shot",
        "latency": latency,
        "prompt": prompt
    }


# ============================================================================
# CHAIN-OF-THOUGHT (CoT) PROMPTING
# ============================================================================

COT_PROMPT = """You are a helpful AI assistant answering questions about university documents.

Context Information:
{context}

Question: {question}

Instructions:
Let's approach this step-by-step:
1. First, identify which parts of the context are relevant to the question
2. Then, extract the specific information needed
3. Finally, synthesize a clear and accurate answer

Reasoning Process:"""


def chain_of_thought_prompting(question: str, context_docs: List[Tuple[Document, float]], 
                               llm=None) -> Dict:
    """
    Chain-of-Thought prompting: Encourages step-by-step reasoning.
    The model explicitly shows its reasoning process before the final answer.
    """
    if llm is None:
        llm = get_llm()
    
    context = format_context(context_docs)
    prompt = COT_PROMPT.format(context=context, question=question)
    
    start_time = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    
    return {
        "answer": response.content,
        "prompt_type": "chain-of-thought",
        "latency": latency,
        "prompt": prompt
    }


# ============================================================================
# FEW-SHOT + CHAIN-OF-THOUGHT COMBINATION
# ============================================================================

FEW_SHOT_COT_PROMPT = """You are a helpful AI assistant answering questions about university documents.

Here are examples showing step-by-step reasoning:

Example 1:
Context: The Final Year Project (FYP) must be submitted by April 30th, 2024. Late submissions will incur a penalty of 5% per day.
Question: What is the FYP submission deadline?
Reasoning:
- Step 1: I found information about FYP submission in the context
- Step 2: The specific deadline mentioned is April 30th, 2024
- Step 3: There's also important information about late submission penalties
Answer: According to the FYP Handbook, the submission deadline is April 30th, 2024. Late submissions will be penalized at 5% per day.

Example 2:
Context: Dr. Atif Tahir is the Head of Computer Science Department. He has published over 50 research papers in AI and Machine Learning.
Question: Who is Dr. Atif Tahir?
Reasoning:
- Step 1: The context provides information about Dr. Atif Tahir's role and achievements
- Step 2: He holds the position of Department Head
- Step 3: He's a researcher with 50+ publications in AI/ML
Answer: Dr. Atif Tahir serves as the Head of the Computer Science Department and is a prolific researcher with over 50 publications in AI and Machine Learning.

Now solve this question using the same step-by-step approach:

Context Information:
{context}

Question: {question}

Reasoning (think step-by-step):"""


def few_shot_cot_prompting(question: str, context_docs: List[Tuple[Document, float]], 
                          llm=None) -> Dict:
    """
    Few-shot + Chain-of-Thought: Combines both techniques.
    Provides examples with explicit reasoning steps.
    Most effective for complex questions requiring multi-step reasoning.
    """
    if llm is None:
        llm = get_llm()
    
    context = format_context(context_docs)
    prompt = FEW_SHOT_COT_PROMPT.format(context=context, question=question)
    
    start_time = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    
    return {
        "answer": response.content,
        "prompt_type": "few-shot-cot",
        "latency": latency,
        "prompt": prompt
    }


# ============================================================================
# ZERO-SHOT CHAIN-OF-THOUGHT
# ============================================================================

ZERO_SHOT_COT_PROMPT = """You are a helpful AI assistant answering questions about university documents.

Context Information:
{context}

Question: {question}

Let's think step by step to answer this question accurately.

Step-by-step reasoning:"""


def zero_shot_cot_prompting(question: str, context_docs: List[Tuple[Document, float]], 
                           llm=None) -> Dict:
    """
    Zero-shot Chain-of-Thought: Uses "Let's think step by step" trigger phrase.
    Encourages reasoning without providing examples.
    """
    if llm is None:
        llm = get_llm()
    
    context = format_context(context_docs)
    prompt = ZERO_SHOT_COT_PROMPT.format(context=context, question=question)
    
    start_time = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    
    return {
        "answer": response.content,
        "prompt_type": "zero-shot-cot",
        "latency": latency,
        "prompt": prompt
    }


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compare_prompting_strategies(question: str, context_docs: List[Tuple[Document, float]],
                                 llm=None, include_prompts: bool = False) -> Dict:
    """Compare prompting strategies on the same question."""
    if llm is None:
        llm = get_llm()

    strategies = [
        ("Zero-Shot", zero_shot_prompting),
        ("Few-Shot", few_shot_prompting),
        ("Chain-of-Thought", chain_of_thought_prompting),
        ("Zero-Shot CoT", zero_shot_cot_prompting),
        ("Few-Shot CoT", few_shot_cot_prompting),
    ]

    results = {}

    for strategy_name, strategy_func in strategies:
        try:
            result = strategy_func(question, context_docs, llm)
            if not include_prompts:
                result = {
                    "answer": result["answer"],
                    "prompt_type": result["prompt_type"],
                    "latency": result["latency"],
                }
            results[strategy_name] = result
        except Exception as exc:
            results[strategy_name] = {"error": str(exc)}

    return results


def save_comparison_results(results: Dict, filepath: str = "prompting_comparison.json"):
    """Save comparison results to JSON"""
    # Remove prompts from saved results (too verbose)
    results_to_save = {}
    for strategy, data in results.items():
        if "error" not in data:
            results_to_save[strategy] = {
                "answer": data["answer"],
                "prompt_type": data["prompt_type"],
                "latency": data["latency"]
            }
        else:
            results_to_save[strategy] = data
    
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)


def _tokenize(text: str) -> set:
    """Tokenize text into lowercase words longer than three characters."""
    return {token for token in re.findall(r"\b\w+\b", text.lower()) if len(token) > 3}


def _context_tokens(context_docs: List[Tuple[Document, float]]) -> set:
    """Build token set from retrieved context documents."""
    tokens = set()
    for doc, _score in context_docs:
        content = doc.page_content or doc.metadata.get("ocr_text", "")
        if content:
            tokens.update(_tokenize(content))
    return tokens


def _context_overlap(answer: str, context_vocab: set) -> float:
    """Compute proportion of answer tokens that appear in the context."""
    answer_tokens = _tokenize(answer)
    if not answer_tokens or not context_vocab:
        return 0.0
    overlap = answer_tokens & context_vocab
    return len(overlap) / len(answer_tokens)


def _quality_label(context_overlap: float, answer_length: int) -> str:
    """Assign a coarse quality label for quick comparison."""
    if answer_length == 0:
        return "no-answer"
    if context_overlap >= 0.35 and answer_length >= 80:
        return "strong"
    if context_overlap >= 0.2:
        return "fair"
    return "weak"


def _summarize_results(results: Dict, context_docs: List[Tuple[Document, float]]) -> List[Dict]:
    """Prepare lightweight summary records for reporting."""
    context_vocab = _context_tokens(context_docs)
    summary = []
    for strategy_name, data in results.items():
        if "error" in data:
            summary.append({
                "strategy": strategy_name,
                "prompt_type": None,
                "latency": None,
                "answer_length": None,
                "context_overlap": None,
                "quality": "error",
                "error": data["error"],
            })
            continue

        answer = data.get("answer", "")
        answer_len = len(answer.split())
        overlap = _context_overlap(answer, context_vocab)
        summary.append({
            "strategy": strategy_name,
            "prompt_type": data.get("prompt_type"),
            "latency": data.get("latency"),
            "answer_length": answer_len,
            "context_overlap": overlap,
            "quality": _quality_label(overlap, answer_len),
            "error": None,
        })
    return summary


def _save_summary_csv(summary: List[Dict], filepath: Path):
    """Persist summary rows to CSV for easy inclusion in reports."""
    if not summary:
        return
    fieldnames = [
        "strategy",
        "prompt_type",
        "latency",
        "answer_length",
        "context_overlap",
        "quality",
        "error",
    ]
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _plot_latency_chart(summary: List[Dict], filepath: Path):
    """Create a bar chart comparing latency across prompting strategies."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    filtered = [row for row in summary if row["latency"] is not None]
    if not filtered:
        return

    strategies = [row["strategy"] for row in filtered]
    latencies = [row["latency"] for row in filtered]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, latencies, color="#4c72b0")
    plt.ylabel("Latency (s)")
    plt.xlabel("Prompting Strategy")
    plt.title("Prompting Strategy Latency Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    for bar, latency in zip(bars, latencies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{latency:.2f}", ha="center", va="bottom", fontsize=9)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def _plot_overlap_chart(summary: List[Dict], filepath: Path):
    """Create a bar chart showing context overlap by strategy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    filtered = [row for row in summary if row["context_overlap"] is not None]
    if not filtered:
        return

    strategies = [row["strategy"] for row in filtered]
    overlaps = [row["context_overlap"] for row in filtered]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, overlaps, color="#55a868")
    plt.ylabel("Context Overlap")
    plt.xlabel("Prompting Strategy")
    plt.title("Context Overlap by Prompting Strategy")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    for bar, overlap in zip(bars, overlaps):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{overlap:.2f}", ha="center", va="bottom", fontsize=9)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def _plot_latency_vs_overlap(summary: List[Dict], filepath: Path):
    """Scatter chart comparing latency against context overlap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    filtered = [row for row in summary if row["latency"] is not None and row["context_overlap"] is not None]
    if not filtered:
        return

    qualities = {row["quality"] for row in filtered if row.get("quality")}
    palette = {
        "strong": "#4c72b0",
        "fair": "#dd8452",
        "weak": "#c44e52",
        "no-answer": "#8172b3",
        "error": "#000000",
    }

    plt.figure(figsize=(8, 5))
    for row in filtered:
        quality = row.get("quality", "fair")
        plt.scatter(row["latency"], row["context_overlap"],
                    color=palette.get(quality, "#55a868"),
                    s=80, edgecolors="black", linewidths=0.6,
                    label=quality if quality not in qualities else None)

    plt.xlabel("Latency (s)")
    plt.ylabel("Context Overlap")
    plt.title("Latency vs. Context Overlap")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Build legend without duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        unique = dict()
        for handle, label in zip(handles, labels):
            if not label or label in unique:
                continue
            unique[label] = handle
        plt.legend(unique.values(), unique.keys(), title="Quality")

    plt.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300)
    plt.close()


def generate_prompting_analysis(question: str, context_docs: List[Tuple[Document, float]],
                                output_dir: str = "prompting_analysis",
                                llm=None, include_prompts: bool = False) -> Dict:
    """Run prompting comparison and export artifacts for reporting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = compare_prompting_strategies(question, context_docs, llm, include_prompts)

    json_path = output_path / "prompting_comparison.json"
    save_comparison_results(results, filepath=str(json_path))

    summary = _summarize_results(results, context_docs)
    _save_summary_csv(summary, output_path / "prompting_summary.csv")
    _plot_latency_chart(summary, output_path / "prompting_latency.png")
    _plot_overlap_chart(summary, output_path / "prompting_overlap.png")
    _plot_latency_vs_overlap(summary, output_path / "prompting_latency_vs_overlap.png")

    return {
        "results": results,
        "summary": summary,
        "artifacts": {
            "json": json_path,
            "csv": output_path / "prompting_summary.csv",
            "latency_chart": output_path / "prompting_latency.png",
            "overlap_chart": output_path / "prompting_overlap.png",
            "latency_vs_overlap": output_path / "prompting_latency_vs_overlap.png",
        },
    }

if __name__ == "__main__":
    from retriever import load_vectorstore, search_text

    vectorstore = load_vectorstore("vectordb/combined_faiss/")
    docs = search_text("What is the FYP report format?", vectorstore, k=3)
    analysis = generate_prompting_analysis(
        question="What is the FYP report format?",
        context_docs=docs,
        output_dir="report_assets"
    )

    print("Prompting analysis artifacts saved:")
    for name, path in analysis["artifacts"].items():
        print(f"  {name}: {path}")