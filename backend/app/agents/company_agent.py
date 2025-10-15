import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# --- Env Variables ---
INDEX_NAME = os.environ.get("INDEX_NAME")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME env variable not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY env variable not set")

# --- Vectorstore Lazy Init ---
def get_vectorstore() -> PineconeVectorStore:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PineconeVectorStore(embedding=embeddings, index_name=INDEX_NAME)

# --- Chat Model (Lazy Loading) ---
chat_model = None

def get_chat_model():
    global chat_model
    if chat_model is None:
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=float(os.getenv("CHAT_TEMP", 0)),
            google_api_key=GEMINI_API_KEY
        )
    return chat_model
    return PineconeVectorStore(embedding=embeddings, index_name=INDEX_NAME)

# --- Chat Model ---
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=float(os.getenv("CHAT_TEMP", 0)),
    google_api_key=GEMINI_API_KEY
)

# --- Company Agent Prompt Template ---
prompt_template = """
You are an **AI-powered Company Intelligence Agent** for **Hero-Vida (Automotive Industry)**.  
Your role is to analyze company-related context and provide **clear, structured, actionable insights**.

### 📝 Guidelines:
- Answer using **only the provided context** (no hallucination).
- If information is missing → reply: "The provided documents do not contain this information."
- Format output in **rich markdown** with headings, bullet points, and tables.
- Focus on **company profiles, EV/ICE offerings, partnerships, product lines, market share, and competitive positioning**.

---

Context:
{context}

Question:
{question}
"""

QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

def format_sources(docs):
    return [
        {"filename": doc.metadata.get("filename","Unknown"),
         "page_or_row": doc.metadata.get("page") or doc.metadata.get("row_index","N/A")}
        for doc in docs[:5]
    ]

# --- QA Chain (Lazy Loading) ---
qa = None

def get_qa():
    global qa
    if qa is None:
        qa = ConversationalRetrievalChain.from_llm(
            llm=get_chat_model(),
            retriever=get_vectorstore().as_retriever(search_kwargs={"k":5}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
    return qa

def run_company_agent(query: str, chat_history: list = None) -> dict:
    """
    Process company-related query and return structured insights + source documents.
    """
    chat_history = chat_history or []
    res = get_qa().invoke({"question": query, "chat_history": chat_history})
    res["source_documents"] = format_sources(res.get("source_documents", []))
    return res
