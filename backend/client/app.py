import streamlit as st
import requests
import shutil
from pathlib import Path
import os
import base64
from PIL import Image
import io

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="EV-GPT BI Multi-Modal", page_icon="⚡", layout="wide")
API_URL = "http://localhost:8001"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------- UTILITY FUNCTIONS ---------------- #
def test_api_connection():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200, resp.json() if resp.status_code == 200 else None
    except:
        return False, None

def send_query(query_text, chat_history=[]):
    try:
        payload = {"query": query_text, "chat_history": chat_history}
        response = requests.post(f"{API_URL}/api/v1/query", json=payload, timeout=240)  # Increased to 4 minutes for first query
        return (response.status_code == 200, response.json() if response.status_code == 200 else f"Error {response.status_code}: {response.text}")
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def trigger_ingestion():
    try:
        payload = {"directory_path": str(DATA_DIR)}
        resp = requests.post(f"{API_URL}/api/v1/ingest", json=payload, timeout=60)
        return (resp.status_code == 200, resp.json() if resp.status_code == 200 else f"Error {resp.status_code}")
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def upload_multimodal_document(file):
    """Upload document for multi-modal processing."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_URL}/api/v1/documents/multimodal", files=files, timeout=120)
        return (response.status_code == 200, response.json() if response.status_code == 200 else f"Error {response.status_code}")
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def query_multimodal_content(query, query_type="auto", top_k=5):
    """Query multi-modal content."""
    try:
        payload = {"query": query, "query_type": query_type, "top_k": top_k}
        response = requests.post(f"{API_URL}/api/v1/query/multimodal", json=payload, timeout=120)
        return (response.status_code == 200, response.json() if response.status_code == 200 else f"Error {response.status_code}")
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def get_multimodal_capabilities():
    """Get multi-modal capabilities."""
    try:
        response = requests.get(f"{API_URL}/api/v1/multimodal/capabilities", timeout=30)
        return (response.status_code == 200, response.json() if response.status_code == 200 else None)
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def save_uploaded_files(uploaded_files):
    saved_files = []
    for file in uploaded_files:
        path = DATA_DIR / file.name
        with open(path, "wb") as f:
            shutil.copyfileobj(file, f)
        saved_files.append(file.name)
    return saved_files

def display_sources(docs):
    for i, doc in enumerate(docs[:3], 1):
        st.markdown(f"**Source {i}: {doc.get('filename','Unknown')} (Page/Row {doc.get('page_or_row','N/A')})**")
        st.markdown(f"_{doc.get('content','No content')[:200]}..._")

def display_multimodal_results(results):
    """Display multi-modal query results with visual indicators."""
    if not results:
        st.info("No results found.")
        return
    
    for i, result in enumerate(results[:5], 1):
        with st.expander(f"📊 Result {i}: {result.get('visual_indicator', 'Content')}"):
            content_type = result.get('content_type_display', 'Text')
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Content Type:** {content_type}")
                st.markdown(result.get('content', 'No content available'))
            
            with col2:
                metadata = result.get('metadata', {})
                if metadata.get('page_number'):
                    st.metric("Page", metadata['page_number'])
                if metadata.get('content_type') == 'table':
                    st.metric("Rows", metadata.get('rows', '?'))
                elif metadata.get('content_type') == 'image':
                    st.metric("Size", f"{metadata.get('width', '?')}x{metadata.get('height', '?')}")

def convert_chat_history(messages, max_history=10):
    history = []
    for i in range(0, len(messages[-max_history:]), 2):
        msgs = messages[-max_history:][i:i+2]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            history.append({"human": msgs[0]["content"], "ai": msgs[1]["content"]})
        elif len(msgs) == 1 and msgs[0]["role"] == "user":
            history.append({"human": msgs[0]["content"], "ai": ""})
    return history

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("🔧 System Status")
    connected, health_data = test_api_connection()
    if connected:
        st.success("✅ API Connected")
        if health_data:
            st.json(health_data)
    else:
        st.error("❌ API Not Connected")
        st.warning("Start backend server:")
        st.code("python main.py")

    # Multi-Modal Capabilities
    st.header("Multi-Modal Features")
    if connected:
        mm_success, mm_caps = get_multimodal_capabilities()
        if mm_success and mm_caps:
            st.success("✅ Multi-Modal Enabled")
            with st.expander("� Capabilities"):
                features = mm_caps.get('features', {})
                for feature, desc in features.items():
                    st.markdown(f"**{feature.replace('_', ' ').title()}**: {desc}")
            
            st.markdown("**Supported Formats:**")
            for fmt in mm_caps.get('supported_formats', []):
                st.write(f"• {fmt}")
        else:
            st.warning("⚠️ Multi-Modal Not Available")

    st.header("📂 Document Management")
    
    # Tab for different upload types
    upload_tab, mm_upload_tab = st.tabs(["📄 Standard", "🎨 Multi-Modal"])
    
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload Documents", type=["pdf","docx","txt","csv"], accept_multiple_files=True
        )
        if uploaded_files:
            saved_files = save_uploaded_files(uploaded_files)
            for f in saved_files:
                st.success(f"✅ Saved: {f}")
            if st.button("🚀 Process Documents"):
                with st.spinner("Processing documents..."):
                    success, result = trigger_ingestion()
                    if success:
                        st.success("✅ Documents processed successfully!")
                        st.json(result)
                    else:
                        st.error(f"❌ Error: {result}")
    
    with mm_upload_tab:
        mm_file = st.file_uploader(
            "Upload for Multi-Modal Analysis", 
            type=["pdf", "docx"], 
            help="Upload PDF or DOCX files for advanced visual and table analysis"
        )
        
        if mm_file:
            if st.button("Process Multi-Modal"):
                with st.spinner("Processing with multi-modal analysis..."):
                    success, result = upload_multimodal_document(mm_file)
                    if success and isinstance(result, dict) and result.get('success'):
                        st.success("✅ Multi-modal processing completed!")
                        stats = result.get('processing_stats', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Images", stats.get('images_analyzed', 0))
                            st.metric("Tables", stats.get('tables_found', 0))
                        with col2:
                            st.metric("Text Chunks", stats.get('text_chunks', 0))
                            st.metric("Total Chunks", stats.get('total_chunks_stored', 0))
                        
                        st.json(result)
                    else:
                        if isinstance(result, dict):
                            error_msg = result.get('error', 'Unknown error')
                        else:
                            error_msg = str(result) if result else 'Upload failed'
                        st.error(f"❌ Error: {error_msg}")

    if st.button("🔄 Process Data Folder"):
        with st.spinner("Processing data folder..."):
            success, result = trigger_ingestion()
            if success:
                st.success("✅ Data folder processed!")
                st.json(result)
            else:
                st.error(f"❌ Error: {result}")

# ---------------- MAIN CHAT ---------------- #
st.title("EV-GPT Business Intelligence")
st.caption("AI-powered insights using your multi-agent RAG system with visual analysis capabilities")

# ---------------- AGENT ROUTING DASHBOARD ---------------- #
with st.expander("🔍 **Agent Routing Dashboard** - Click to see LangChain orchestration", expanded=False):
    st.markdown("### 🎯 Available Agents & Keywords")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛒 **Sales Agent**")
        st.info("**Keywords:** sales, sell, buy, purchase, dealer, showroom, booking, delivery")
        st.markdown("**Sample Query:** *'Where can I buy a Hero Vida scooter in Mumbai?'*")
        
        st.markdown("#### 💰 **Pricing Agent**") 
        st.info("**Keywords:** price, cost, pricing, expensive, cheap, affordable, budget")
        st.markdown("**Sample Query:** *'What is the price of Hero Vida Pro?'*")
    
    with col2:
        st.markdown("#### 🏢 **Company Agent**")
        st.info("**Keywords:** hero, vida, bajaj, tvs, company, business, revenue, earnings")
        st.markdown("**Sample Query:** *'Tell me about Hero MotoCorp financial performance'*")
        
        st.markdown("#### 🎨 **Multimodal Agent**")
        st.info("**Keywords:** image, chart, graph, visual, picture, diagram, table")
        st.markdown("**Sample Query:** *'Show me a chart of electric scooter sales'*")

    st.markdown("### 📊 **Test These Queries (Based on Your Data):**")
    test_queries = [
        "🛒 Where can I purchase Hero Vida V1 scooter?",
        "💰 What is the price of Hero Vida V1 Plus and V1 Pro?", 
        "🏢 Tell me about Hero MotoCorp electric vehicle business",
        "🎨 Show me Hero Vida specifications table",
        "💰 Compare Hero Vida V1 Plus vs V1 Pro pricing",
        "🏢 What are Hero company's EV manufacturing details?",
        "🛒 What warranty does Hero provide for Vida scooters?"
    ]
    
    for query in test_queries:
        if st.button(f"Test: {query}", key=f"test_{hash(query)}"):
            st.session_state.test_query = query[2:]  # Remove emoji

# Query mode selector
query_mode = st.radio(
    "Query Mode:",
    ["Smart Agent (Auto-Routing)", "Multi-Modal Search", "Direct RAG"],
    horizontal=True,
    help="Choose how to process your query"
)

# Multi-Modal Query Options
if query_mode == "Multi-Modal Search":
    col1, col2 = st.columns([2, 1])
    with col1:
        query_type = st.selectbox(
            "Analysis Type:",
            ["auto", "visual", "tabular", "financial", "technical"],
            help="Specify the type of content to focus on"
        )
    with col2:
        top_k = st.slider("Max Results", 1, 10, 5)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            
            # Handle multi-modal results
            if metadata.get("query_type") and "results" in metadata:
                st.markdown("### Multi-Modal Results")
                display_multimodal_results(metadata["results"])
            else:
                # Enhanced agent routing display in chat history
                agent_used = metadata.get('agent_used', 'Unknown')
                intent = metadata.get('intent', 'Unknown')
                
                agent_colors = {
                    'Sales Agent': '🛒',
                    'Pricing Agent': '💰',
                    'Company Agent': '🏢', 
                    'Multimodal Agent': '🎨',
                    'General QA Agent': '🤖',
                    'RAG Agent': '📚'
                }
                
                agent_icon = agent_colors.get(agent_used, '❓')
                
                # Enhanced display for chat history  
                if 'Sales' in agent_used:
                    st.markdown(f"<div style='background-color: #e8f5e8; border-radius: 5px; padding: 8px; margin: 5px 0;'><b>🛒 {agent_used}</b> | Intent: {intent}</div>", unsafe_allow_html=True)
                elif 'Pricing' in agent_used:
                    st.markdown(f"<div style='background-color: #fff3cd; border-radius: 5px; padding: 8px; margin: 5px 0;'><b>💰 {agent_used}</b> | Intent: {intent}</div>", unsafe_allow_html=True)
                elif 'Company' in agent_used:
                    st.markdown(f"<div style='background-color: #e2e3f3; border-radius: 5px; padding: 8px; margin: 5px 0;'><b>🏢 {agent_used}</b> | Intent: {intent}</div>", unsafe_allow_html=True)
                elif 'Multimodal' in agent_used:
                    st.markdown(f"<div style='background-color: #f8d7da; border-radius: 5px; padding: 8px; margin: 5px 0;'><b>🎨 {agent_used}</b> | Intent: {intent}</div>", unsafe_allow_html=True)
                else:
                    st.info(f"**{agent_icon} {agent_used}** | Intent: `{intent}`")
                if metadata.get("source_documents"):
                    with st.expander("📑 Sources"):
                        display_sources(metadata["source_documents"])

# Handle test queries
if hasattr(st.session_state, 'test_query'):
    prompt = st.session_state.test_query
    del st.session_state.test_query
    st.rerun()

# Chat input
prompt_placeholder = {
    "🤖 Smart Agent (Auto-Routing)": "Ask about sales, pricing, company, or describe visual content...",
    "🎨 Multi-Modal Search": "Search for charts, tables, images, or visual data...",
    "📊 Direct RAG": "Ask any question from the documents..."
}

# Live routing prediction (for testing purposes)
if query_mode == "🤖 Smart Agent (Auto-Routing)":
    with st.container():
        test_input = st.text_input("🔍 **Test Query Routing** (type to see which agent would handle it):", key="route_test")
        if test_input:
            # Simple client-side prediction based on keywords
            test_lower = test_input.lower()
            
            if any(word in test_lower for word in ['image', 'chart', 'graph', 'visual', 'picture', 'diagram']):
                st.success("🎨 **Predicted Route:** Multimodal Agent")
            elif any(word in test_lower for word in ['hero', 'vida', 'bajaj', 'tvs', 'company', 'business', 'revenue']):
                st.success("🏢 **Predicted Route:** Company Agent") 
            elif any(word in test_lower for word in ['price', 'cost', 'pricing', 'expensive', 'cheap', 'affordable']):
                st.success("💰 **Predicted Route:** Pricing Agent")
            elif any(word in test_lower for word in ['sales', 'sell', 'buy', 'purchase', 'dealer', 'showroom']):
                st.success("🛒 **Predicted Route:** Sales Agent")
            else:
                st.info("🤖 **Predicted Route:** General QA Agent")

prompt = None
if hasattr(st.session_state, 'test_query'):
    prompt = st.session_state.test_query
    del st.session_state.test_query
else:
    prompt = st.chat_input(prompt_placeholder.get(query_mode, "Ask anything..."))

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if connected:
            with st.spinner("Processing..."):
                success = False
                response = None
                
                if query_mode == "🎨 Multi-Modal Search":
                    # Multi-modal query
                    success, response = query_multimodal_content(prompt, query_type, top_k)
                    if success and response.get('success'):
                        answer = f"Found {response.get('total_results', 0)} multi-modal results for your query."
                        st.markdown(answer)
                        
                        # Display results
                        st.markdown("### 🎨 Multi-Modal Results")
                        display_multimodal_results(response.get('results', []))
                        
                        # Store in session with special metadata
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "metadata": {
                                "query_type": response.get('query_type'),
                                "results": response.get('results', []),
                                "total_results": response.get('total_results', 0),
                                "mode": "multimodal"
                            }
                        })
                    else:
                        error_msg = response.get('error', 'Multi-modal search failed') if response else 'Query failed'
                        st.error(f"❌ {error_msg}")
                        st.session_state.messages.append({"role":"assistant","content":f"❌ {error_msg}"})
                
                elif query_mode == "📊 Direct RAG":
                    # Direct RAG query (bypass agents)
                    # This would call a direct RAG endpoint if you have one
                    st.info("Direct RAG mode - would bypass agents for basic document search")
                    success = False
                    response = "Direct RAG mode not yet implemented"
                
                else:
                    # Smart Agent (Auto-Routing) - existing functionality
                    chat_history = convert_chat_history(st.session_state.messages)
                    success, response = send_query(prompt, chat_history)
                    
                    if success:
                        answer = response.get("answer","No answer received")
                        st.markdown(answer)
                        
                        # Enhanced agent routing display
                        agent_used = response.get('agent_used', 'Unknown')
                        intent = response.get('intent', 'Unknown')
                        
                        # Color coding for different agents
                        agent_colors = {
                            'Sales Agent': '🛒',
                            'Pricing Agent': '💰', 
                            'Company Agent': '🏢',
                            'Multimodal Agent': '🎨',
                            'General QA Agent': '🤖',
                            'RAG Agent': '📚'
                        }
                        
                        agent_icon = agent_colors.get(agent_used, '❓')
                        
                        # Make the routing VERY visible
                        st.success(f"**🎯 ORCHESTRATION SUCCESS: {agent_icon} {agent_used}**")
                        
                        # Create a prominent colored box
                        if 'Sales' in agent_used:
                            st.markdown(f"<div style='background-color: #e8f5e8; border-left: 5px solid #28a745; padding: 10px; margin: 10px 0;'><b>🛒 SALES AGENT ACTIVATED</b><br>Intent: {intent}<br>Query routed successfully!</div>", unsafe_allow_html=True)
                        elif 'Pricing' in agent_used:
                            st.markdown(f"<div style='background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin: 10px 0;'><b>💰 PRICING AGENT ACTIVATED</b><br>Intent: {intent}<br>Query routed successfully!</div>", unsafe_allow_html=True)
                        elif 'Company' in agent_used:
                            st.markdown(f"<div style='background-color: #e2e3f3; border-left: 5px solid #6f42c1; padding: 10px; margin: 10px 0;'><b>� COMPANY AGENT ACTIVATED</b><br>Intent: {intent}<br>Query routed successfully!</div>", unsafe_allow_html=True)
                        elif 'Multimodal' in agent_used:
                            st.markdown(f"<div style='background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 10px; margin: 10px 0;'><b>🎨 MULTIMODAL AGENT ACTIVATED</b><br>Intent: {intent}<br>Query routed successfully!</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color: #d1ecf1; border-left: 5px solid #17a2b8; padding: 10px; margin: 10px 0;'><b>🤖 {agent_used.upper()}</b><br>Intent: {intent}<br>Query processed!</div>", unsafe_allow_html=True)
                        
                        if response.get("source_documents"):
                            with st.expander("📑 Sources"):
                                display_sources(response["source_documents"])
                        
                        st.session_state.messages.append({"role":"assistant","content":answer,"metadata":response})
                    else:
                        st.error(f"❌ {response}")
                        st.session_state.messages.append({"role":"assistant","content":f"❌ {response}"})
                
        else:
            st.error("❌ API not connected. Please start the backend server.")
            st.session_state.messages.append({"role":"assistant","content":"❌ API not connected."})


st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("🚀 **Backend:** main.py")
with col2:
    st.markdown("🤖 **Agents:** agents/")
with col3:
    st.markdown("📊 **RAG:** stateful_bot.py")
with col4:
    st.markdown("🎨 **Multi-Modal:** vision AI + tables")
