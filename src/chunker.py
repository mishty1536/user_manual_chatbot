"""Token-based text chunking with overlap."""
print("  Loading chunker.py...", flush=True)

import tiktoken
from typing import List

from .models import ChunkData
from .config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

print("  Loading tiktoken encoding (may download on first run)...", flush=True)
# Use cl100k_base encoding (compatible with modern models)
_encoding = tiktoken.get_encoding("cl100k_base")
print("  Tiktoken encoding loaded!", flush=True)


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_encoding.encode(text))


def chunk_text(
    text: str,
    source_filename: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS
) -> List[ChunkData]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        source_filename: Source document filename
        chunk_size: Target tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of ChunkData objects
    """
    if not text.strip():
        return []

    import sys
    print(f"  [Chunker] Tokenizing {source_filename} ({len(text)} chars)...", flush=True)
    # Encode entire text to tokens
    tokens = _encoding.encode(text)
    total_tokens = len(tokens)
    print(f"  [Chunker] {source_filename}: {total_tokens} tokens", flush=True)

    if total_tokens <= chunk_size:
        # Text fits in a single chunk
        return [ChunkData(
            text=text,
            chunk_index=0,
            source_filename=source_filename,
            start_char=0,
            end_char=len(text)
        )]

    chunks = []
    chunk_index = 0
    start_token = 0
    current_char_pos = 0  # Track character position incrementally
    estimated_chunks = (total_tokens // (chunk_size - overlap)) + 1
    print(f"  [Chunker] Starting chunking loop (est. {estimated_chunks} chunks)...", flush=True)

    while start_token < total_tokens:
        print(f"  [Chunker] Creating chunk {chunk_index + 1}...", flush=True)
        # Calculate end token for this chunk
        end_token = min(start_token + chunk_size, total_tokens)

        # Decode tokens back to text
        chunk_tokens = tokens[start_token:end_token]
        chunk_text_str = _encoding.decode(chunk_tokens)

        # Use incrementally tracked character position
        start_char = current_char_pos
        end_char = start_char + len(chunk_text_str)

        chunks.append(ChunkData(
            text=chunk_text_str,
            chunk_index=chunk_index,
            source_filename=source_filename,
            start_char=start_char,
            end_char=end_char
        ))

        chunk_index += 1

        # If we've reached the end of the document, we're done
        if end_token >= total_tokens:
            break

        # Move start position, accounting for overlap
        next_start_token = end_token - overlap

        # Calculate character offset for the overlap
        if next_start_token < total_tokens:
            # Decode only the non-overlapping portion to update char position
            non_overlap_tokens = tokens[start_token:next_start_token]
            non_overlap_text = _encoding.decode(non_overlap_tokens)
            current_char_pos += len(non_overlap_text)

        start_token = next_start_token

    print(f"  [Chunker] Done! Created {len(chunks)} chunks", flush=True)
    return chunks


def chunk_document(document_text: str, source_filename: str) -> List[ChunkData]:
    """
    Chunk a document's text.

    Args:
        document_text: Full text of the document
        source_filename: Source document filename

    Returns:
        List of ChunkData objects
    """
    return chunk_text(document_text, source_filename)
