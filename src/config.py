"""Configuration constants for the PDF Embeddings application."""

# Chunking settings
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_PERCENT = 0.10  # 10%
CHUNK_OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * CHUNK_OVERLAP_PERCENT)  # 50 tokens preserves continuity 

# Embedding settings
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768 #vector size 
EMBEDDING_INPUT_LIMIT = 2048  # tokens per text

# Generation settings
GENERATION_MODEL = "gemini-3-flash-preview"

# Batch settings
EMBEDDING_BATCH_SIZE = 100  # Max chunks per batch request

# Query settings
TOP_K_CHUNKS = 5 #controls how much context is used 

# UI settings
SHOW_BUNDLE_DETAILS = False  # Show/hide "Bundle Details" section by default
SHOW_SOURCE_CHUNKS = False   # Show/hide "Source Chunks" section by default

# File limits
MAX_PDF_FILES = 15
MAX_PDF_SIZE_MB = 50

# Document summary prompt (one per PDF, prepended to all chunks)
DOCUMENT_SUMMARY_PROMPT = """<document>
{full_document}
</document>

Please provide a brief summary (2-3 sentences) of this document that describes:
1. What the document is about
2. The main topics or themes covered

This summary will be prepended to text chunks for better search retrieval. Answer only with the summary and nothing else."""

# Answer generation prompt
ANSWER_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.

Context from relevant document chunks:
{chunks_context}

User Question: {query}

Please provide a comprehensive answer based on the context above.

IMPORTANT: In your response:
1. Clearly state which chapter(s) and section(s) contain the answer (e.g., "According to Chapter 3, Section 2.1...")
2. If the information spans multiple sections, mention all relevant locations
3. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available

Format your response with the source location first, then the answer."""

WARRANTY_PROMPT_TEMPLATE = """
You are a helpful assistant answering warranty and service-related questions
based ONLY on the provided document context.

Context from the document:
{chunks_context}

User Question:
{query}

Instructions (VERY IMPORTANT):
- Use ONLY the information available in the context.
- Do NOT assume or invent any data.
- For each component, identify:
  1. The warranty or replacement frequency (if specified).
  2. The inspection interval or range (if inspection is mentioned).

Output rules:
- Present the result STRICTLY in a table with THREE columns:

| Component / System | Warranty / Replacement Frequency | Operation |

- If a replacement interval (km/years) is mentioned, include it under "Warranty / Replacement Frequency" and set Operation = "Replacement".
- If a component is marked for inspection, include the inspection interval or range
  (e.g., "1,000 km to 150,000 km at periodic intervals") under
  "Warranty / Replacement Frequency" and set Operation = "Inspection only".
- If inspection includes additional actions (cleaning, lubrication, software reset),
  mention them under Operation.
- Do NOT leave the frequency column empty if an interval or range is available.
- Do NOT add explanations outside the table.
"""

