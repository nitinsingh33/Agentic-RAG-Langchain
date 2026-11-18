"""Hero Motocorp agentic document research workspace."""

import os

import requests
import streamlit as st


API_ROOT = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")
API_BASE_URL = f"{API_ROOT}/api/v1"
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "csv", "xlsx", "xls"]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024))) // (1024 * 1024)

st.set_page_config(
    page_title="EV Moto GPT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --mint:#9AF0BE; --mint-bright:#C1FFD7; --ink:#ECF7F0; --muted:#A5B8AD; --line:rgba(184,226,198,.14); --panel:rgba(17,34,26,.78); }
      .stApp { color:var(--ink); background:radial-gradient(ellipse at 78% 5%,rgba(57,143,91,.22),transparent 34%),radial-gradient(ellipse at 6% 70%,rgba(33,102,68,.18),transparent 38%),linear-gradient(135deg,#07100D 0%,#0B1712 48%,#102219 100%); background-attachment:fixed; }
      [data-testid="stHeader"] { background:rgba(7,16,13,.72); color:var(--ink); backdrop-filter:blur(14px); }
      [data-testid="stHeader"] button { color:var(--ink); }
      [data-testid="stHeader"] svg { fill:var(--ink); }
      section.main > div.block-container { max-width:1380px; padding-top:2rem; padding-bottom:3rem; }
      [data-testid="stSidebar"] { background:linear-gradient(165deg,rgba(16,34,25,.98),rgba(8,18,14,.98)); border-right:1px solid var(--line); }
      [data-testid="stSidebar"] * { color:#E5F2E9; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted); }
      .hero-masthead { display:flex; align-items:center; gap:13px; margin:.2rem 0 1.6rem; }
      .hero-mark { display:grid; place-items:center; width:46px; height:46px; border-radius:15px; background:linear-gradient(145deg,#C1FFD7,#65D997); color:#10251A; font-weight:900; font-size:18px; letter-spacing:-.06em; box-shadow:0 0 28px rgba(125,226,163,.25); }
      .hero-brand { color:var(--ink); font-size:24px; font-weight:800; letter-spacing:.015em; }
      .hero-brand span { color:var(--mint); }
      .hero-kicker { color:#9EB7A7; font-size:11px; letter-spacing:.14em; font-weight:750; text-transform:uppercase; }
      .hero-title { color:#F0FBF3; font-size:37px; line-height:1.13; font-weight:740; letter-spacing:-.035em; margin:.25rem 0 .55rem; }
      .hero-subtitle { color:#AEC0B4; font-size:15px; line-height:1.6; margin-bottom:1.5rem; max-width:900px; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(24,47,35,.9),rgba(13,28,21,.88)); border:1px solid var(--line); padding:13px 15px; border-radius:14px; box-shadow:0 8px 28px rgba(0,0,0,.16); transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
      [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:rgba(154,240,190,.38); box-shadow:0 12px 30px rgba(0,0,0,.22),0 0 18px rgba(126,226,163,.06); }
      [data-testid="stTabs"] [role="tablist"] { gap:1.25rem; border-bottom:1px solid var(--line); }
      [data-testid="stTabs"] button[role="tab"] { color:#A7B8AE; font-weight:650; transition:color .18s ease; }
      [data-testid="stTabs"] button[role="tab"]:hover { color:var(--mint-bright); }
      [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--mint); }
      [data-testid="stAlert"] { border-radius:13px; border:1px solid rgba(154,240,190,.2); background:linear-gradient(120deg,rgba(28,62,42,.78),rgba(16,37,26,.78)); color:#DDF5E4; }
      [data-testid="stFileUploader"] section { background:linear-gradient(145deg,rgba(20,42,30,.86),rgba(12,27,20,.86)); border:1px dashed rgba(154,240,190,.34); border-radius:15px; transition:border-color .18s ease,background .18s ease; }
      [data-testid="stFileUploader"] section:hover { border-color:var(--mint); background:linear-gradient(145deg,rgba(27,57,39,.94),rgba(14,34,23,.94)); }
      [data-testid="stChatInput"] textarea { background:rgba(14,29,21,.94); color:var(--ink); border:1px solid rgba(184,226,198,.22); border-radius:15px; }
      [data-testid="stChatInput"] textarea:focus { border-color:var(--mint); box-shadow:0 0 0 1px var(--mint),0 0 22px rgba(143,230,178,.12); }
      .stButton > button { border-radius:12px; font-weight:620; min-height:44px; transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease,background .18s ease; }
      .stButton > button[kind="secondary"] { background:linear-gradient(135deg,rgba(24,45,33,.95),rgba(14,30,22,.95)); color:#E4F3E9; border:1px solid rgba(184,226,198,.18); }
      .stButton > button[kind="secondary"]:hover { color:#EFFFF3; border-color:rgba(154,240,190,.65); background:linear-gradient(135deg,rgba(37,78,52,.96),rgba(19,49,32,.96)); box-shadow:0 7px 22px rgba(0,0,0,.22),0 0 18px rgba(143,230,178,.11); transform:translateY(-2px); }
      .stButton > button[kind="primary"] { background:linear-gradient(115deg,#9AF0BE,#68D99B); color:#10251A; border:1px solid #9AF0BE; box-shadow:0 4px 18px rgba(111,220,154,.16); }
      .stButton > button[kind="primary"]:hover { background:linear-gradient(115deg,#C1FFD7,#85EBAE); border-color:#C1FFD7; box-shadow:0 8px 24px rgba(111,220,154,.26); transform:translateY(-2px); }
      [data-testid="stExpander"] { background:linear-gradient(145deg,rgba(19,39,28,.86),rgba(12,26,19,.86)); border:1px solid var(--line); border-radius:13px; transition:border-color .18s ease; }
      [data-testid="stExpander"]:hover { border-color:rgba(154,240,190,.34); }
      [data-testid="stChatMessage"] { background:rgba(17,35,25,.54); border:1px solid rgba(184,226,198,.1); border-radius:16px; }
      .research-callout { padding:15px 18px; border:1px solid rgba(154,240,190,.2); background:linear-gradient(110deg,rgba(28,62,42,.78),rgba(14,33,23,.82)); border-radius:14px; color:#DDF5E4; margin:.4rem 0 1rem; }
      @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition:none !important; scroll-behavior:auto !important; } }
      @media (max-width:700px) { .hero-title { font-size:30px; } .hero-subtitle { font-size:14px; } }
    </style>
    <div class="hero-masthead">
      <div class="hero-mark">EV</div>
      <div><div class="hero-brand">EV MOTO <span>GPT</span></div>
      <div class="hero-kicker">Hero Motocorp intelligence</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def check_backend():
    response = requests.get(f"{API_ROOT}/ready", timeout=8)
    response.raise_for_status()
    return response.json()


def upload_document(uploaded_file):
    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def ask_question(question: str, top_k: int, mode: str):
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"query": question, "top_k": top_k, "mode": mode},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_research_result(result: dict):
    st.markdown(result.get("answer", ""))

    metrics = result.get("metrics", {})
    if metrics:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Manager", f"{metrics.get('planning_ms', 0):g} ms")
        metric_cols[1].metric("Retrieval", f"{metrics.get('retrieval_ms', 0):g} ms", f"{metrics.get('retrieved_chunks', 0)} chunks")
        expert_label = "Experts" if result.get("mode") == "agentic" else "Generation"
        metric_cols[2].metric(expert_label, f"{metrics.get('agent_ms', 0):g} ms")
        metric_cols[3].metric("Synthesis", f"{metrics.get('synthesis_ms', 0):g} ms")
        metric_cols[4].metric("End to end", f"{metrics.get('total_ms', 0):g} ms")

    agent_steps = result.get("agents", [])
    if agent_steps:
        trace_title = (
            "Agent run · manager → retrieval → specialists"
            if result.get("mode") == "agentic"
            else "Single-pass baseline run"
        )
        with st.expander(trace_title, expanded=False):
            for step in agent_steps:
                with st.container(border=True):
                    st.markdown(f"**{step.get('name', 'Agent')}** · {step.get('role', '')} · {step.get('status', 'unknown')}")
                    st.write(step.get("output", ""))

    sources = result.get("sources", [])
    if sources:
        st.markdown("#### Evidence")
        for source in sources:
            source_id = source.get("source_id", "S?")
            filename = source.get("filename", "Unknown document")
            location = source.get("page")
            if location is None:
                location = source.get("row_index")
            suffix = f" · page/row {location}" if location is not None else ""
            score = source.get("score")
            similarity = f" · similarity {score:.3f}" if isinstance(score, (int, float)) else ""
            with st.expander(f"[{source_id}]  {filename}{suffix}{similarity}"):
                st.write(source.get("content", ""))
                st.caption(f"Type: {source.get('source_type') or 'unknown'} · Chunk: {source.get('chunk_id', '—')}")


with st.sidebar:
    st.markdown("### EV Moto GPT")
    mode_label = st.radio(
        "Answer workflow",
        ["Agentic team", "Single-pass baseline"],
        help="Use the baseline to compare a single general RAG call with manager-led specialist research.",
    )
    mode = "agentic" if mode_label == "Agentic team" else "baseline"
    top_k = st.slider("Evidence passages", min_value=1, max_value=10, value=5)
    if mode == "agentic":
        st.caption("The manager routes questions to focused experts. All answers use your indexed documents.")
    else:
        st.caption("One general RAG call, included as a baseline for comparison. Answers use your indexed documents.")
    st.divider()

    st.markdown("### Service status")
    if st.button("Check connection", use_container_width=True):
        try:
            health = check_backend()
            st.session_state.backend_status = (True, health.get("status", "ready"))
        except requests.RequestException as exc:
            st.session_state.backend_status = (False, str(exc))

    status = st.session_state.get("backend_status")
    if status is None:
        st.caption("Connection not checked yet")
    elif status[0]:
        st.success("API ready")
    else:
        st.error("API unavailable")

    st.divider()
    st.caption(f"Vector index · {os.getenv('INDEX_NAME', 'ev-gpt-rag')}")
    st.caption(f"Model · {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.research_history = []
        st.rerun()


st.markdown('<div class="hero-kicker">Internal research workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Your EV knowledge,<br>ready for deeper questions.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">EV Moto GPT coordinates focused business experts over your Hero Motocorp documents, with source evidence for every answer.</div>', unsafe_allow_html=True)

ask_tab, knowledge_tab = st.tabs(["Research desk", "Knowledge base"])

with ask_tab:
    if "research_history" not in st.session_state:
        st.session_state.research_history = []

    if not st.session_state.research_history:
        st.markdown(
            "<div class='research-callout'>"
            "Start with a business question. The manager will select a focused specialist or coordinate several for a cross-functional answer.</div>",
            unsafe_allow_html=True,
        )
        sample_questions = [
            "What product launches and features are described in the documents?",
            "Summarize the sales performance by region and identify trends.",
            "Compare the available model pricing and variants.",
        ]
        prompt_cols = st.columns(3)
        for index, sample in enumerate(sample_questions):
            if prompt_cols[index].button(sample, key=f"sample-{index}", use_container_width=True):
                st.session_state.pending_question = sample
                st.rerun()

    for item in st.session_state.research_history:
        with st.chat_message(item["role"]):
            if item["role"] == "user":
                st.write(item["content"])
            else:
                render_research_result(item["result"])

    question = st.chat_input("Ask about Hero Motocorp products, sales, pricing, strategy…")
    pending_question = st.session_state.pop("pending_question", None)
    question = pending_question or question
    if question:
        try:
            with st.spinner("Manager routing · retrieving evidence · consulting specialists…"):
                result = ask_question(question.strip(), top_k, mode)
            st.session_state.research_history.append({"role": "user", "content": question.strip()})
            st.session_state.research_history.append({"role": "assistant", "result": result})
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Research request failed: {exc}")

with knowledge_tab:
    st.markdown("### Add internal documents")
    st.markdown("Index reports, product information, sales data, and other business documents for the research agents.")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_TYPES,
        help=f"Supported: PDF, DOCX, TXT, Markdown, CSV, XLSX, XLS · up to {MAX_UPLOAD_MB} MB.",
    )
    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name} · {uploaded_file.size / (1024 * 1024):.1f} MB")
        if st.button("Index document", type="primary", disabled=uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024):
            try:
                with st.spinner("Extracting, chunking, and indexing this document…"):
                    result = upload_document(uploaded_file)
                st.success(f"Indexed {result.get('filename', uploaded_file.name)}")
                summary_cols = st.columns(3)
                summary_cols[0].metric("Chunks created", result.get("chunks_created", 0))
                summary_cols[1].metric("Embedding", f"{result.get('metrics', {}).get('embedding_ms', 0):g} ms")
                summary_cols[2].metric("Index update", f"{result.get('metrics', {}).get('upsert_ms', 0):g} ms")
            except requests.RequestException as exc:
                st.error(f"Indexing failed: {exc}")
    st.markdown("---")
    st.markdown("**Supported evidence formats** · PDF · Word · Markdown · text · CSV · Excel")
