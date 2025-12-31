"""Data models for the PDF Embeddings application."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class DocumentData:
    """Represents an extracted PDF document."""
    filename: str
    full_text: str
    page_texts: List[str]
    pdf_bytes: bytes


@dataclass
class ChunkData:
    """Represents a text chunk before contextualization."""
    text: str
    chunk_index: int
    source_filename: str
    start_char: int
    end_char: int


@dataclass
class ContextualizedChunk:
    """Chunk with prepended context, ready for embedding."""
    original_text: str
    context: str
    contextualized_text: str  # context + original_text
    chunk_index: int
    source_filename: str


@dataclass
class EmbeddingsBundle:
    """Complete embeddings data to be pickled."""
    chunks: List[ContextualizedChunk]
    embeddings: List[List[float]]  # Parallel to chunks
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "creation_date": datetime.now().isoformat(),
                "total_chunks": len(self.chunks),
            }


@dataclass
class QueryResult:
    """Result of a query operation."""
    query: str
    answer: str
    relevant_chunks: List[ContextualizedChunk]
    similarity_scores: List[float]
    source_filenames: List[str]
