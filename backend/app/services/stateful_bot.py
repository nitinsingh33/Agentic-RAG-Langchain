import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from ..utils.logger import logger

# Load environment variables
load_dotenv()


def get_vectorstore() -> PineconeVectorStore:
    """Get initialized Pinecone vector store"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = PineconeVectorStore(
            index_name=os.environ.get("INDEX_NAME"),
            embedding=embeddings,
            pinecone_api_key=os.environ.get("PINECONE_API_KEY")
        )
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise


def get_qa() -> ConversationalRetrievalChain:
    """Get QA chain with memory"""
    try:
        vectorstore = get_vectorstore()
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.7
        )
        
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True
        )
        
        return qa_chain
    
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise