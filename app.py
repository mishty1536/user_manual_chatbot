"""
PDF Embeddings Application
A Streamlit app for creating contextual embeddings from PDFs and querying them.
"""

print("Loading app.py...", flush=True) #Confirms that app.py execution has started

from datetime import datetime #loads datetime class into memory used for ZIP filename generation and bundle creation timestamp

import os
import streamlit as st #loads streamlit framework used for UI
from google import genai #loads gemini SDK  used for API key validation, creating embeddings, for answer generation 

print("Importing src modules...", flush=True) #internal modules are being loaded helps debug delays or crashes 

from src.chunker import chunk_document #checks if src.chunker is loaded or not if not it opens the file and executes from top to bottom 
from src.config import ( #similary checks the laading of file 
    CHUNK_OVERLAP_TOKENS, #50
    CHUNK_SIZE_TOKENS, #500
    EMBEDDING_DIMENSIONS, #768 vector size
    EMBEDDING_MODEL, #gemini-embedding-001
    MAX_PDF_FILES, #15
    MAX_PDF_SIZE_MB, #50 file limits
    SHOW_SOURCE_CHUNKS, # Show/hide "Source Chunks" section by default
)
from src.context_generator import ContextGenerationError, contextualize_chunks #checks if the file is loaded or not if not executes from top to bottom
from src.embeddings import EmbeddingError, create_embeddings_batch
from src.file_handler import ( #for saving and loading embedding 
    BundleError,
    create_embeddings_bundle,
    load_embeddings_bundle,
    validate_zip_bundle,
)
from src.models import EmbeddingsBundle #for structured data flow 
from src.pdf_processor import PDFProcessingError, extract_text_from_uploaded_files #extract text form pdfs
from src.query_engine import QueryError, process_query #handles query -> answers pipeline

#UI 
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "client": None,
        "embeddings_bundle": None,
        "pdf_files": None,
        "processing_complete": False,
        "last_query": None,
        "last_answer": None,
        "last_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("API key not configured. Please contact administrator.")
        st.stop()
    return genai.Client(api_key=api_key)
    
def render_create_embeddings_tab():
    """Render the Create Embeddings tab."""
    st.header("Create Embeddings from PDFs")

    if not st.session_state.client:
        st.info("Please enter your API key in the sidebar to continue.")
        return

    uploaded_files = st.file_uploader(
        f"Upload PDF files (max {MAX_PDF_FILES})",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        # Validate file count
        if len(uploaded_files) > MAX_PDF_FILES:
            st.error(
                f"Maximum {MAX_PDF_FILES} files allowed. You uploaded {len(uploaded_files)}."
            )
            return

        # Validate file sizes
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            if size_mb > MAX_PDF_SIZE_MB:
                st.error(f"File '{f.name}' exceeds {MAX_PDF_SIZE_MB}MB limit.")
                return

        st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
        for f in uploaded_files:
            st.caption(f"- {f.name} ({f.size / 1024:.1f} KB)")

        if st.button("Create Embeddings", type="primary"):
            process_pdfs(uploaded_files)


def process_pdfs(uploaded_files):
    """Process uploaded PDFs and create embeddings."""
    print("\n" + "=" * 50, flush=True)
    print("STARTING PDF PROCESSING", flush=True)
    print("=" * 50, flush=True)

    client = st.session_state.client
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Extract text from PDFs
        print("\n[Step 1/4] Extracting text from PDFs...", flush=True)
        status_text.text("Step 1/4: Extracting text from PDFs...")
        progress_bar.progress(5)

        documents = extract_text_from_uploaded_files(uploaded_files) #from pdf-processor
        print(f"[Step 1/4] Extracted {len(documents)} documents", flush=True)
        for doc in documents:
            print(f"  - {doc.filename}: {len(doc.full_text)} chars", flush=True)
        progress_bar.progress(15)

        # Step 2: Chunk all documents
        print("\n[Step 2/4] Chunking documents...", flush=True)
        status_text.text("Step 2/4: Chunking documents...")
        all_chunks = []
        doc_chunks_map = {}  # Map document to its chunks

        for doc in documents:
            print(f"  Processing: {doc.filename}", flush=True)
            chunks = chunk_document(doc.full_text, doc.filename) #from chunker
            print(f"  -> Created {len(chunks)} chunks", flush=True)
            doc_chunks_map[doc.filename] = (doc, chunks)
            all_chunks.extend(chunks)

        st.write(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")
        progress_bar.progress(25)

        # Step 3: Generate context for each chunk
        status_text.text(
            "Step 3/4: Generating context for chunks (this may take a while)..."
        )
        all_contextualized = []
        total_chunks = len(all_chunks)
        chunks_processed = 0

        for filename, (doc, chunks) in doc_chunks_map.items():

            def update_progress(current, total):
                nonlocal chunks_processed
                chunks_processed = (
                    sum(
                        len(c)
                        for f, (_, c) in list(doc_chunks_map.items())[
                            : list(doc_chunks_map.keys()).index(filename)
                        ]
                    )
                    + current
                )
                pct = 25 + int((chunks_processed / total_chunks) * 40)
                progress_bar.progress(min(pct, 65))
                status_text.text(
                    f"Step 3/4: Generating context... ({chunks_processed}/{total_chunks} chunks)"
                )

            contextualized = contextualize_chunks(doc, chunks, client, update_progress) #from context_generation 
            all_contextualized.extend(contextualized)

        progress_bar.progress(65)

        # Step 4: Create embeddings
        status_text.text("Step 4/4: Creating embeddings...")
        texts_to_embed = [c.contextualized_text for c in all_contextualized]

        def embedding_progress(current, total):
            pct = 65 + int((current / total) * 30)
            progress_bar.progress(min(pct, 95))
            status_text.text(f"Step 4/4: Creating embeddings... ({current}/{total})")

        embeddings = create_embeddings_batch( #from embeddings
            texts_to_embed,
            client,
            progress_callback=embedding_progress,
        )

        progress_bar.progress(95)

        # Create bundle
        status_text.text("Creating download bundle...")
        bundle = EmbeddingsBundle(
            chunks=all_contextualized,
            embeddings=embeddings,
            metadata={
                "creation_date": datetime.now().isoformat(),
                "total_chunks": len(all_contextualized),
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "source_files": [d.filename for d in documents],
            },
        )

        # Collect PDF bytes
        pdf_files = {doc.filename: doc.pdf_bytes for doc in documents}

        # Create zip
        zip_bytes = create_embeddings_bundle(bundle, pdf_files)

        progress_bar.progress(100)
        status_text.text("Complete!")

        st.success(
            f"Successfully processed {len(documents)} PDFs into {len(all_contextualized)} chunks!"
        )

        # Download button
        st.download_button(
            label="Download Embeddings Bundle",
            data=zip_bytes,
            file_name=f"embeddings_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            type="primary",
        )

    except PDFProcessingError as e:
        st.error(f"PDF Processing Error: {str(e)}")
    except ContextGenerationError as e:
        st.error(f"Context Generation Error: {str(e)}")
    except EmbeddingError as e:
        st.error(f"Embedding Error: {str(e)}")
    except BundleError as e:
        st.error(f"Bundle Creation Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


def clear_chat():
    """Clear the chat (question and answer) but keep the bundle."""
    st.session_state.last_query = None
    st.session_state.last_answer = None
    st.session_state.last_result = None


def render_query_tab():
    """Render the Query tab with two-column layout."""
    st.header("Query Your Documents")
    

    # Check if bundle is already loaded
    bundle_loaded = st.session_state.embeddings_bundle is not None

    if not bundle_loaded:
        # ============ UPLOAD VIEW (Full width when no bundle) ============
        st.subheader("Load Embeddings Bundle")

        uploaded_zip = st.file_uploader(
            "Upload embeddings bundle (.zip)",
            type="zip",
            key="zip_uploader",
        )

        if uploaded_zip:
            zip_bytes = uploaded_zip.read()
            uploaded_zip.seek(0)

            with st.spinner("Loading embeddings bundle..."):
                is_valid, error_msg = validate_zip_bundle(zip_bytes)
                if not is_valid:
                    st.error(f"Invalid bundle: {error_msg}")
                    return

                try:
                    bundle, pdf_files = load_embeddings_bundle(zip_bytes)
                    st.session_state.embeddings_bundle = bundle
                    st.session_state.pdf_files = pdf_files
                    st.session_state.last_zip_name = uploaded_zip.name
                    clear_chat()
                    st.rerun()
                except BundleError as e:
                    st.error(f"Failed to load bundle: {str(e)}")
                    return
    else:
        # ============ FULL-WIDTH Q&A (After bundle loaded) ============
        bundle = st.session_state.embeddings_bundle
        pdf_files = st.session_state.pdf_files

        # Top bar with bundle popover and change button
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            with st.popover("Bundle"):
                st.markdown(
                    f"**{len(bundle.chunks)} chunks** from {len(pdf_files)} PDFs"
                )
                st.divider()
                for f in bundle.metadata.get("source_files", []):
                    st.caption(f"• {f}")
        with col2:
            if st.button("Change", use_container_width=True):
                st.session_state.embeddings_bundle = None
                st.session_state.pdf_files = None
                clear_chat()
                st.rerun()

        # Full-width Q&A
        query = st.text_input(
            "Enter your question:",
            key="query_input",
            placeholder="Ask something about your documents...",
        )

        # Buttons row
        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 6])
        with btn_col1:
            search_clicked = st.button(
                "Search & Answer",
                type="primary",
                use_container_width=True,
            )
        with btn_col2:
            if st.button("New Chat", use_container_width=True):
                clear_chat()
                st.rerun()

        if search_clicked:
            if len(query.strip()) < 2:
                st.warning("Please enter at least 2 characters.")
            else:
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = process_query(
                            query,
                            bundle,
                            pdf_files,
                            st.session_state.client,
                        )
                        st.session_state.last_query = query
                        st.session_state.last_answer = result.answer
                        st.session_state.last_result = result
                    except QueryError as e:
                        st.error(f"Query Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")

        # Display answer
        if st.session_state.last_answer:
            st.divider()
            st.markdown("### Answer")
            st.markdown(st.session_state.last_answer)

            result = st.session_state.last_result
            if result and SHOW_SOURCE_CHUNKS:
                with st.expander("Source Chunks", expanded=True):
                    for i, (chunk, score) in enumerate(
                        zip(result.relevant_chunks, result.similarity_scores), 1
                    ):
                        st.markdown(
                            f"**Chunk {i}** (Score: {score:.3f}) - {chunk.source_filename}"
                        )
                        st.text(
                            chunk.original_text[:300]
                            + ("..." if len(chunk.original_text) > 300 else "")
                        )
                        if i < len(result.relevant_chunks):
                            st.divider()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="MG Windsor Chatbot",
        page_icon="📄",
        layout="wide",
    )

    init_session_state()

    if st.session_state.client is None:
        st.session_state.client = get_genai_client()
        
    st.title("Know Your car - AI Assistant")
    st.caption("Powered by AI.")

    #Hide create embeddings tab 
    tab1 = st.tabs(["Query"])[0]

    # Main tabs
    #tab1, tab2 = st.tabs(["Create Embeddings", "Query"])

    #with tab1:
      #  render_create_embeddings_tab()

    #with tab2:
        #render_query_tab()


if __name__ == "__main__":
    main()

