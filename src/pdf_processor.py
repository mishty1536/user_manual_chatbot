"""PDF text extraction using PyMuPDF."""

import fitz  # PyMuPDF
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile

from .models import DocumentData


class PDFProcessingError(Exception):
    """Error during PDF processing."""
    pass


def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> DocumentData:
    """
    Extract text from PDF bytes.

    Args:
        pdf_bytes: Raw PDF file bytes
        filename: Original filename for reference

    Returns:
        DocumentData with full text and per-page text

    Raises:
        PDFProcessingError: If PDF cannot be processed
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            page_texts.append(text)

        full_text = "\n\n".join(page_texts)
        doc.close()

        if not full_text.strip():
            raise PDFProcessingError(
                f"No extractable text found in '{filename}'. "
                "The PDF may be scanned or image-based."
            )

        return DocumentData(
            filename=filename,
            full_text=full_text,
            page_texts=page_texts,
            pdf_bytes=pdf_bytes
        )

    except fitz.FileDataError as e:
        raise PDFProcessingError(f"Invalid PDF file '{filename}': {str(e)}")
    except Exception as e:
        raise PDFProcessingError(f"Error processing '{filename}': {str(e)}")


def extract_text_from_uploaded_files(
    uploaded_files: List[UploadedFile]
) -> List[DocumentData]:
    """
    Process multiple uploaded PDF files.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        List of DocumentData objects

    Raises:
        PDFProcessingError: If any PDF cannot be processed
    """
    documents = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        # Reset file pointer for potential re-reads
        uploaded_file.seek(0)

        doc_data = extract_text_from_pdf(pdf_bytes, uploaded_file.name)
        documents.append(doc_data)

    return documents
