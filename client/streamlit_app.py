import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8001/api/v1"

st.set_page_config(
    page_title="EV-GPT",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# API Client
# --------------------------------------------------

def check_backend():
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def upload_document(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files=files,
        timeout=300,
    )

    response.raise_for_status()
    return response.json()


def ask_question(query, top_k):
    payload = {
        "query": query,
        "top_k": top_k,
    }

    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 EV-GPT")
st.caption("AI-powered Retrieval-Augmented Generation over your documents")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("System")

    if st.button("Check Backend", use_container_width=True):
        try:
            health = check_backend()

            if health.get("status") == "ok":
                st.success("Backend is running")
            else:
                st.warning("Backend responded, but status is not OK.")

        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

    st.divider()

    st.header("Retrieval")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5,
    )


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

st.header("📄 Upload Documents")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt", "md", "csv", "xlsx", "xls"],
)

if uploaded_file is not None:

    if st.button("Upload & Index", type="primary"):

        with st.spinner("Uploading and indexing document..."):

            try:
                result = upload_document(uploaded_file)

                st.success(
                    f"Successfully indexed `{result['filename']}`"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Documents",
                        result.get("documents_loaded", 0),
                    )

                with col2:
                    st.metric(
                        "Chunks",
                        result.get("chunks_created", 0),
                    )

                with col3:
                    st.metric(
                        "Index",
                        result.get("index_name", "N/A"),
                    )

            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")


st.divider()


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

st.header("💬 Ask your documents")

query = st.text_area(
    "Question",
    placeholder="Ask something about your uploaded documents...",
    height=120,
)

if st.button("Ask", type="primary", use_container_width=True):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            try:
                result = ask_question(
                    query=query.strip(),
                    top_k=top_k,
                )

                st.subheader("Answer")

                st.write(result.get("answer", ""))

                sources = result.get("sources", [])

                if sources:

                    st.divider()
                    st.subheader("📚 Sources")

                    for index, source in enumerate(sources, start=1):

                        filename = source.get(
                            "filename",
                            "Unknown",
                        )

                        page = source.get("page")

                        content = source.get(
                            "content",
                            "",
                        )

                        title = f"Source {index}: {filename}"

                        if page is not None:
                            title += f" — Page {page}"

                        with st.expander(title):

                            st.write(content)

                            st.caption(
                                f"Source type: "
                                f"{source.get('source_type', 'unknown')} "
                                f"| Chunk: "
                                f"{source.get('chunk_id', 'unknown')}"
                            )

            except requests.RequestException as exc:
                st.error(f"Query failed: {exc}")

            except Exception as exc:
                st.error(f"Unexpected error: {exc}")