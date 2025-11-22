"""
Generator Module
================
Uses LLM to generate answers based on retrieved context from RAG system.
"""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


def get_api_key():
    """Get Groq API key from environment"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return api_key


# Initialize LLM
try:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=2048,
        api_key=get_api_key(),
    )
except Exception as e:
    print(f"[WARNING] Failed to initialize Groq LLM: {e}")
    llm = None


# RAG Prompt Template
RAG_PROMPT = """You are a helpful AI assistant answering questions based on the provided context from university documents.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say "I don't have enough information in the provided context to answer this question."
3. Be concise and specific
4. If referencing specific information, mention which document it came from
5. For numerical data or specific facts, quote them exactly as they appear

Answer:"""


def format_context(results: List[Tuple[Document, float]], include_scores=False) -> str:
    """
    Format retrieved documents into context string

    Args:
        results: List of (Document, score) tuples from retriever
        include_scores: Whether to include similarity scores

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, (doc, score) in enumerate(results, 1):
        # Extract source filename
        source = doc.metadata.get("source", "unknown")
        source_file = source.split("/")[-1].replace("_chunks.json", "")

        # Get content
        content = (
            doc.page_content if doc.page_content else doc.metadata.get("ocr_text", "")
        )

        # Format context entry
        score_info = f" (Relevance: {score:.3f})" if include_scores else ""
        context_entry = f"[Source {i}: {source_file}{score_info}]\n{content}\n"
        context_parts.append(context_entry)

    return "\n".join(context_parts)


def generate_answer(
    question: str,
    context_docs: List[Tuple[Document, float]],
    model_name="llama-3.1-70b-versatile",
    temperature=0.2,
) -> dict:
    """
    Generate answer using LLM based on retrieved context

    Args:
        question: User's question
        context_docs: List of (Document, score) tuples from retriever
        model_name: Groq model to use
        temperature: LLM temperature (0-1, lower = more focused)

    Returns:
        dict with 'answer', 'context', and 'sources'
    """
    if not llm:
        return {
            "answer": "Error: LLM not initialized. Please set GROQ_API_KEY environment variable.",
            "context": "",
            "sources": [],
        }

    # Format context
    context = format_context(context_docs, include_scores=False)

    # Create prompt
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    messages = prompt.format_messages(context=context, question=question)

    # Generate answer
    try:
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # Extract sources
    sources = []
    for doc, score in context_docs:
        source_file = (
            doc.metadata.get("source", "unknown")
            .split("/")[-1]
            .replace("_chunks.json", "")
        )
        sources.append(
            {
                "file": source_file,
                "type": doc.metadata.get("type", "unknown"),
                "score": float(score),
            }
        )

    return {"answer": answer, "context": context, "sources": sources}


def generate_answer_with_streaming(
    question: str, context_docs: List[Tuple[Document, float]]
):
    """
    Generate answer with streaming output

    Args:
        question: User's question
        context_docs: List of (Document, score) tuples from retriever

    Yields:
        Chunks of the generated answer
    """
    if not llm:
        yield "Error: LLM not initialized. Please set GROQ_API_KEY environment variable."
        return

    # Format context
    context = format_context(context_docs, include_scores=False)

    # Create prompt
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    messages = prompt.format_messages(context=context, question=question)

    # Generate answer with streaming
    try:
        for chunk in llm.stream(messages):
            yield chunk.content
    except Exception as e:
        yield f"Error generating answer: {str(e)}"


def print_rag_response(question: str, response: dict):
    """
    Pretty print RAG response

    Args:
        question: Original question
        response: Response dict from generate_answer
    """
    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 70}\n")

    print("ANSWER:")
    print(response["answer"])

    print(f"\n{'=' * 70}")
    print("SOURCES:")
    print(f"{'=' * 70}")
    for i, source in enumerate(response["sources"], 1):
        print(
            f"{i}. {source['file']} (Type: {source['type']}, Score: {source['score']:.3f})"
        )
    print()


# ============================================================================
# CLI INTERFACE
# ============================================================================
if __name__ == "__main__":
    import sys
    from retriever import load_vectorstore, search_text, print_results

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python generator.py 'your question'")
        print("\nExample:")
        print("  python generator.py 'What is the FYP report format?'")
        sys.exit(1)

    # Get question
    question = " ".join(sys.argv[1:])

    # Load vectorstore
    print("[1/3] Loading vectorstore...")
    vectorstore = load_vectorstore()

    # Retrieve relevant documents
    print(f"[2/3] Retrieving relevant documents for: {question}")
    results = search_text(question, vectorstore, k=5)
    print_results(results)
    # Generate answer
    print("[3/3] Generating answer...")
    response = generate_answer(question, results)

    # Print response
    print_rag_response(question, response)
