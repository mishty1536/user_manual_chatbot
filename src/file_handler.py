"""Zip file creation/extraction and pickle serialization."""

import io
import pickle
import zipfile
from typing import Dict, Tuple

from .models import EmbeddingsBundle


class BundleError(Exception):
    """Error with embeddings bundle creation or loading."""
    pass


def create_embeddings_bundle(
    embeddings_bundle: EmbeddingsBundle,
    pdf_files: Dict[str, bytes]
) -> bytes:
    """
    Create a zip file containing embeddings and original PDFs.

    Args:
        embeddings_bundle: EmbeddingsBundle with chunks and embeddings
        pdf_files: Dict mapping filename to PDF bytes

    Returns:
        Zip file as bytes
    """
    try:
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add embeddings pickle
            embeddings_bytes = pickle.dumps(embeddings_bundle)
            zf.writestr('embeddings.pkl', embeddings_bytes)

            # Add PDFs in pdfs/ folder
            for filename, pdf_bytes in pdf_files.items():
                zf.writestr(f'pdfs/{filename}', pdf_bytes)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    except Exception as e:
        raise BundleError(f"Failed to create bundle: {str(e)}")


def load_embeddings_bundle(
    zip_bytes: bytes
) -> Tuple[EmbeddingsBundle, Dict[str, bytes]]:
    """
    Load and validate a zip bundle.

    Args:
        zip_bytes: Zip file bytes

    Returns:
        Tuple of (EmbeddingsBundle, dict of PDF files)

    Raises:
        BundleError: If zip structure is invalid
    """
    try:
        zip_buffer = io.BytesIO(zip_bytes)

        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            # Validate structure
            namelist = zf.namelist()

            if 'embeddings.pkl' not in namelist:
                raise BundleError("Missing embeddings.pkl in zip bundle")

            # Load embeddings
            embeddings_bytes = zf.read('embeddings.pkl')
            embeddings_bundle = pickle.loads(embeddings_bytes)

            if not isinstance(embeddings_bundle, EmbeddingsBundle):
                raise BundleError("Invalid embeddings data format")

            # Load PDFs
            pdf_files = {}
            for name in namelist:
                if name.startswith('pdfs/') and name != 'pdfs/':
                    filename = name[5:]  # Remove 'pdfs/' prefix
                    pdf_files[filename] = zf.read(name)

            if not pdf_files:
                raise BundleError("No PDF files found in bundle")

            return embeddings_bundle, pdf_files

    except zipfile.BadZipFile:
        raise BundleError("Invalid zip file")
    except pickle.UnpicklingError:
        raise BundleError("Corrupted embeddings data")
    except Exception as e:
        if isinstance(e, BundleError):
            raise
        raise BundleError(f"Failed to load bundle: {str(e)}")


def validate_zip_bundle(zip_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate uploaded zip bundle structure without fully loading it.

    Args:
        zip_bytes: Zip file bytes

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        zip_buffer = io.BytesIO(zip_bytes)

        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            namelist = zf.namelist()

            if 'embeddings.pkl' not in namelist:
                return False, "Missing embeddings.pkl in zip bundle"

            has_pdfs = any(n.startswith('pdfs/') and n != 'pdfs/' for n in namelist)
            if not has_pdfs:
                return False, "No PDF files found in pdfs/ folder"

            return True, ""

    except zipfile.BadZipFile:
        return False, "Invalid zip file"
    except Exception as e:
        return False, f"Error validating bundle: {str(e)}"
