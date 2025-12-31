"""Gemini embedding API wrapper with batching and retry logic."""

import time
import re
from typing import List, Callable, Optional
from google import genai
from google.genai import types

from .config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE


class EmbeddingError(Exception):
    """Error during embedding creation."""
    pass


def extract_retry_delay(error_message: str) -> Optional[float]:
    """Extract retry delay from error message if present."""
    patterns = [
        r'retry in ([\d.]+)s',
        r'retryDelay[\'"]?\s*[:=]\s*[\'"]?([\d.]+)s?[\'"]?',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def call_with_retry(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 10.0,
    max_delay: float = 120.0,
    context: str = ""
):
    """
    Call a function with exponential backoff retry on rate limit errors.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            last_error = e

            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                if attempt < max_retries:
                    suggested_delay = extract_retry_delay(error_str)

                    if suggested_delay:
                        delay = min(suggested_delay + 5, max_delay)
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)

                    print(f"\n    ⚠️  [RATE LIMIT] Hit API rate limit! ({context})", flush=True)
                    print(f"    ⚠️  Waiting {delay:.0f} seconds before retry {attempt + 1}/{max_retries}...", flush=True)

                    # Countdown display
                    for remaining in range(int(delay), 0, -10):
                        print(f"    ⏳ {remaining}s remaining...", flush=True)
                        time.sleep(min(10, remaining))

                    print(f"    ✅ Retrying now...\n", flush=True)
                    continue

            raise

    raise EmbeddingError(f"Failed after {max_retries} retries: {last_error}")


def truncate_text_to_token_limit(text: str, max_tokens: int = 2048) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated text
    """
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def create_embedding(
    text: str,
    client: genai.Client,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS
) -> List[float]:
    """
    Create embedding for a single text.

    Args:
        text: Text to embed
        client: Gemini client instance
        model: Embedding model name
        dimensions: Output embedding dimensions

    Returns:
        Embedding vector as list of floats
    """
    # Truncate if necessary
    text = truncate_text_to_token_limit(text)

    def make_request():
        result = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=dimensions)
        )
        return result.embeddings[0].values

    try:
        return call_with_retry(make_request, context="Embedding")
    except Exception as e:
        raise EmbeddingError(f"Failed to create embedding: {str(e)}")


def create_embeddings_batch(
    texts: List[str],
    client: genai.Client,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Create embeddings for multiple texts with batching.

    Args:
        texts: List of texts to embed
        client: Gemini client instance
        model: Embedding model name
        dimensions: Output embedding dimensions
        batch_size: Number of texts per API call
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of embedding vectors
    """
    # Truncate all texts first
    print(f"    [Embeddings] Preparing {len(texts)} texts...", flush=True)
    truncated_texts = [truncate_text_to_token_limit(t) for t in texts]

    all_embeddings = []
    total = len(truncated_texts)
    num_batches = (total + batch_size - 1) // batch_size

    print(f"    [Embeddings] Creating embeddings in {num_batches} batches...", flush=True)

    try:
        for batch_num, i in enumerate(range(0, total, batch_size), 1):
            batch = truncated_texts[i:i + batch_size]

            def make_batch_request(b=batch):
                result = client.models.embed_content(
                    model=model,
                    contents=b,
                    config=types.EmbedContentConfig(output_dimensionality=dimensions)
                )
                return result

            print(f"    [Embeddings] Batch {batch_num}/{num_batches} ({len(batch)} texts)...", flush=True)
            result = call_with_retry(make_batch_request, context=f"Batch {batch_num}/{num_batches}")

            for emb in result.embeddings:
                all_embeddings.append(emb.values)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        print(f"    [Embeddings] Done! Created {len(all_embeddings)} embeddings", flush=True)
        return all_embeddings

    except Exception as e:
        raise EmbeddingError(f"Failed to create embeddings batch: {str(e)}")
