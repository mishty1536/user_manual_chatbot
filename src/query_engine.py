"""
Query processing and answer generation.
"""

from typing import Dict, List
from google import genai

from .models import EmbeddingsBundle, ContextualizedChunk, QueryResult
from .embeddings import create_embedding
from .similarity import find_top_k_similar
from .config import (
    ANSWER_PROMPT_TEMPLATE,
    WARRANTY_PROMPT_TEMPLATE,
    GENERATION_MODEL,
    TOP_K_CHUNKS
)


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
    Unified RAG pipeline:
    same retrieval, prompt decides output format.
    """

    try:
        # 1. Embed query
        query_embedding = create_embedding(query, client)

        # 2. Similarity search
        similar_results = find_top_k_similar(
            query_embedding,
            embeddings_bundle.embeddings,
            k=top_k
        )

        # 3. Collect chunks
        relevant_chunks: List[ContextualizedChunk] = []
        similarity_scores = []
        source_filenames = set()

        for idx, score in similar_results:
            chunk = embeddings_bundle.chunks[idx]
            relevant_chunks.append(chunk)
            similarity_scores.append(score)
            source_filenames.add(chunk.source_filename)

        # 4. Build context
        chunks_context = ""
        for i, chunk in enumerate(relevant_chunks, 1):
            chunks_context += f"\n--- Chunk {i} (from {chunk.source_filename}) ---\n"
            chunks_context += f"{chunk.original_text}\n"

        # 5. Detect warranty / service intent
        warranty_keywords = [
            "warranty",
            "replacement",
            "inspection",
            "service",
            "interval"
        ]

        is_warranty_query = any(
            keyword in query.lower()
            for keyword in warranty_keywords
        )

        # 6. Choose prompt
        if is_warranty_query:
            prompt = WARRANTY_PROMPT_TEMPLATE.format(
                chunks_context=chunks_context,
                query=query
            )
        else:
            prompt = ANSWER_PROMPT_TEMPLATE.format(
                chunks_context=chunks_context,
                query=query
            )

        # 7. Generate answer
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        return QueryResult(
            query=query,
            answer=response.text.strip(),
            relevant_chunks=relevant_chunks,
            similarity_scores=similarity_scores,
            source_filenames=list(source_filenames)
        )

    except Exception as e:
        raise QueryError(f"Query processing failed: {str(e)}")
