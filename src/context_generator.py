"""Document summary generation using Gemini (one summary per PDF)."""

import time
import re
from typing import List, Callable, Optional
from google import genai

from .models import DocumentData, ChunkData, ContextualizedChunk
from .config import DOCUMENT_SUMMARY_PROMPT, GENERATION_MODEL


class ContextGenerationError(Exception):
    """Error during context generation."""
    pass


def extract_retry_delay(error_message: str) -> Optional[float]:
    """Extract retry delay from error message if present."""
    # Look for patterns like "retry in 52.475473228s" or "retryDelay': '52s'"
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
    max_delay: float = 120.0
):
    """
    Call a function with exponential backoff retry on rate limit errors.

    Args:
        func: Function to call (should take no arguments)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Result from the function

    Raises:
        ContextGenerationError: If all retries exhausted
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            last_error = e

            # Check if it's a rate limit error
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                if attempt < max_retries:
                    # Try to extract suggested delay from error
                    suggested_delay = extract_retry_delay(error_str)

                    if suggested_delay:
                        delay = min(suggested_delay + 5, max_delay)  # Add 5s buffer
                    else:
                        # Exponential backoff: 10, 20, 40, 80, 120 seconds
                        delay = min(base_delay * (2 ** attempt), max_delay)

                    print(f"\n    ⚠️  [RATE LIMIT] Hit API rate limit!", flush=True)
                    print(f"    ⚠️  Waiting {delay:.0f} seconds before retry {attempt + 1}/{max_retries}...", flush=True)

                    # Countdown display
                    for remaining in range(int(delay), 0, -10):
                        print(f"    ⏳ {remaining}s remaining...", flush=True)
                        time.sleep(min(10, remaining))

                    print(f"    ✅ Retrying now...\n", flush=True)
                    continue

            # Not a rate limit error, or we've exhausted retries
            raise

    raise ContextGenerationError(f"Failed after {max_retries} retries: {last_error}")


def generate_document_summary(
    full_document: str,
    client: genai.Client,
    model: str = GENERATION_MODEL
) -> str:
    """
    Generate a summary for the entire document.

    Args:
        full_document: The full document text
        client: Gemini client instance
        model: Model to use for generation

    Returns:
        Summary string to prepend to all chunks from this document
    """
    # For very large documents, truncate to avoid token limits
    max_doc_chars = 100000  # ~25k tokens approximately
    if len(full_document) > max_doc_chars:
        # Keep beginning and end of document
        half = max_doc_chars // 2
        full_document = (
            full_document[:half] +
            "\n\n[... document truncated for length ...]\n\n" +
            full_document[-half:]
        )

    prompt = DOCUMENT_SUMMARY_PROMPT.format(full_document=full_document)

    def make_request():
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()

    try:
        print(f"    [Context] Generating document summary...", flush=True)
        summary = call_with_retry(make_request)
        print(f"    [Context] Summary generated ({len(summary)} chars)", flush=True)
        return summary
    except Exception as e:
        raise ContextGenerationError(f"Failed to generate document summary: {str(e)}")


def contextualize_chunks(
    document: DocumentData,
    chunks: List[ChunkData],
    client: genai.Client,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[ContextualizedChunk]:
    """
    Process all chunks for a document, adding the document summary as context.

    This generates ONE summary per document and prepends it to ALL chunks
    from that document.

    Args:
        document: The source document
        chunks: List of chunks to contextualize
        client: Gemini client instance
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of ContextualizedChunk objects
    """
    # Generate ONE summary for the entire document
    summary = generate_document_summary(document.full_text, client)

    # Apply the same summary to all chunks
    contextualized = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        # Prepend summary to chunk
        contextualized_text = f"[Document: {document.filename}]\n{summary}\n\n{chunk.text}"

        ctx_chunk = ContextualizedChunk(
            original_text=chunk.text,
            context=summary,
            contextualized_text=contextualized_text,
            chunk_index=chunk.chunk_index,
            source_filename=chunk.source_filename
        )
        contextualized.append(ctx_chunk)

        if progress_callback:
            progress_callback(i + 1, total)

    print(f"    [Context] Applied summary to {len(chunks)} chunks", flush=True)
    return contextualized
