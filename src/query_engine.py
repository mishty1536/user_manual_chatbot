"""Query processing and answer generation."""

from typing import Dict, List
from google import genai

from .models import EmbeddingsBundle, ContextualizedChunk, QueryResult
from .embeddings import create_embedding
from .similarity import find_top_k_similar
from .config import ANSWER_PROMPT_TEMPLATE, GENERATION_MODEL, TOP_K_CHUNKS


class QueryError(Exception):
    """Error during query processing."""
    pass


def process_query(
    query: str,
    embeddings_bundle: EmbeddingsBundle,
    pdf_files: Dict[str, bytes],
    client: genai.Client,
    top_k: int = TOP_K_CHUNKS
) -> QueryResult:
    """
    Full query processing pipeline.

    Args:
        query: User's question
        embeddings_bundle: Loaded EmbeddingsBundle
        pdf_files: Dict of PDF files
        client: Gemini client instance
        top_k: Number of top chunks to retrieve

    Returns:
        QueryResult with answer and relevant chunks
    """
    try:
        # Step 1: Create embedding for query
        query_embedding = create_embedding(query, client)

        # Step 2: Find most similar chunks
        similar_results = find_top_k_similar(
            query_embedding,
            embeddings_bundle.embeddings,
            k=top_k
        )

        # Step 3: Get the relevant chunks
        relevant_chunks = []
        similarity_scores = []
        source_filenames = set()

        for idx, score in similar_results:
            chunk = embeddings_bundle.chunks[idx]
            relevant_chunks.append(chunk)
            similarity_scores.append(score)
            source_filenames.add(chunk.source_filename)

        # Step 4: Generate answer
        answer = generate_answer(query, relevant_chunks, client)

        return QueryResult(
            query=query,
            answer=answer,
            relevant_chunks=relevant_chunks,
            similarity_scores=similarity_scores,
            source_filenames=list(source_filenames)
        )

    except Exception as e:
        raise QueryError(f"Query processing failed: {str(e)}")


def generate_answer(
    query: str,
    relevant_chunks: List[ContextualizedChunk],
    client: genai.Client,
    model: str = GENERATION_MODEL
) -> str:
    """
    Generate answer using Gemini with context from relevant chunks.

    Args:
        query: User's question
        relevant_chunks: List of relevant contextualized chunks
        client: Gemini client instance
        model: Model to use for generation

    Returns:
        Generated answer string
    """
    # Build context from chunks
    chunks_context = ""
    for i, chunk in enumerate(relevant_chunks, 1):
        chunks_context += f"\n--- Chunk {i} (from {chunk.source_filename}) ---\n"
        chunks_context += f"Context: {chunk.context}\n"
        chunks_context += f"Content: {chunk.original_text}\n"

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        chunks_context=chunks_context,
        query=query
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        raise QueryError(f"Answer generation failed: {str(e)}")
