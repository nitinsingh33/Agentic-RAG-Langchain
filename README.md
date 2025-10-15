# Agentic RAG System for Business Intelligence

A production-ready, multi-agent Retrieval-Augmented Generation (RAG) system designed for intelligent document analysis and business intelligence in the automotive/EV industry. The system leverages specialized AI agents, multi-modal processing capabilities, and advanced semantic search to provide contextual, accurate responses from complex business documents.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Agent Specifications](#agent-specifications)
- [Multi-Modal Processing](#multi-modal-processing)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an enterprise-grade RAG system that combines the power of Large Language Models (LLMs) with specialized domain agents to analyze business documents, extract insights, and provide intelligent responses. The system is particularly optimized for the electric vehicle (EV) and automotive industry, handling diverse document types including PDFs, DOCX, CSV, Excel files, and multi-modal content.

### Problem Statement

Organizations struggle to efficiently extract actionable insights from large volumes of unstructured business documents. Traditional keyword-based search systems fail to understand context and semantic meaning, while manual document analysis is time-consuming and error-prone.

### Solution

An intelligent, multi-agent RAG system that:
- Understands natural language queries and routes them to specialized agents
- Processes multi-modal content (text, images, tables, charts)
- Maintains conversational context across interactions
- Provides source-attributed, accurate responses
- Scales efficiently with document volume

## Key Features

### 1. Multi-Agent Architecture
- **Router Agent**: Intelligent query classification and routing using semantic similarity
- **Company Agent**: Specialized in company profiles, partnerships, and competitive analysis
- **Sales Agent**: Focuses on sales data, KPIs, regional performance, and market trends
- **Pricing Agent**: Handles pricing strategies, cost analysis, and competitive pricing

### 2. Multi-Modal Document Processing
- Text extraction from PDFs, DOCX, TXT, and Markdown files
- Table extraction and analysis from structured documents
- Image analysis using Google Gemini Vision API
- Chart and graph interpretation
- OCR capabilities for scanned documents

### 3. Advanced RAG Pipeline
- Semantic document chunking with configurable parameters
- Vector embeddings using Sentence Transformers
- Pinecone vector database for efficient similarity search
- Conversational memory with context preservation
- Source attribution for transparency and verification

### 4. Production-Ready Backend
- FastAPI-based REST API with comprehensive endpoints
- Asynchronous processing for optimal performance
- Robust error handling and logging
- CORS support for cross-origin requests
- Health check and monitoring endpoints

### 5. Interactive Frontend
- Streamlit-based user interface
- Real-time query processing
- Multi-modal search capabilities
- Document upload and ingestion
- Chat history preservation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Streamlit Web Interface                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Router Agent (Intent Detection)         │   │
│  └───────┬──────────────────┬───────────────┬───────────┘   │
│          │                  │               │               │
│  ┌───────▼────┐   ┌────────▼─────┐   ┌────▼──────────┐      │
│  │  Company   │   │    Sales     │   │    Pricing    │      │
│  │   Agent    │   │    Agent     │   │     Agent     │      │
│  └────────────┘   └──────────────┘   └───────────────┘      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Processing & Storage Layer                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │ Multi-Modal  │  │   Vector DB    │  │   LLM Service  │   │
│  │  Processor   │  │   (Pinecone)   │  │ (Gemini 1.5)   │   │
│  └──────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **LangChain**: LLM orchestration and RAG pipeline
- **FastAPI**: High-performance web framework
- **Streamlit**: Interactive web interface

### AI/ML Components
- **Google Gemini 1.5 Pro**: Primary LLM for text generation and vision tasks
- **Sentence Transformers**: Embedding generation (all-MiniLM-L6-v2)
- **Pinecone**: Vector database for semantic search
- **HuggingFace**: Model hub and embeddings

### Document Processing
- **PyPDF**: PDF text extraction
- **python-docx**: Microsoft Word document processing
- **openpyxl**: Excel file handling
- **Pillow (PIL)**: Image processing

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nitinsingh33/Agentic-RAG-Langchain.git
cd Agentic-RAG-Langchain
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install client dependencies
cd client
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
INDEX_NAME=your_index_name
PINECONE_HOST=your_pinecone_host_url

# Multi-Modal Configuration
MULTIMODAL_ENABLED=true
VISION_MODEL=gemini-1.5-pro
OCR_ENABLED=true
TABLE_EXTRACTION_ENABLED=true

# Server Configuration
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Frontend Configuration
STREAMLIT_PORT=8501
BACKEND_URL=http://localhost:8000
```

## Configuration

### Document Chunking Parameters

Edit `backend/app/services/ingest.py`:

```python
CHUNK_SIZE = 2000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
```

### Agent Intent Threshold

Edit `backend/app/agents/router_agent.py`:

```python
INTENT_THRESHOLD = 0.35  # Minimum confidence for agent selection
```

### Embedding Model

Edit agent files to change the embedding model:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Usage

### Starting the Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Starting the Streamlit Client

```bash
cd backend/client
streamlit run app.py --server.port 8501
```

The web interface will be available at `http://localhost:8501`

### Document Ingestion

#### Via Web Interface
1. Navigate to the Streamlit interface
2. Use the "Upload Documents" section in the sidebar
3. Select files (PDF, DOCX, TXT, MD, CSV, XLSX)
4. Click "Process Data Folder" to ingest all documents

#### Via API
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "data"}'
```

#### Programmatically

```python
from app.services.ingest import ingest_directory

result = ingest_directory("path/to/documents")
print(f"Processed {result['files_processed']} files")
```

### Querying the System

#### Via Web Interface
1. Enter your query in the chat input
2. Select query mode (Smart Agent, Multi-Modal, Direct RAG)
3. Review the response and source documents

#### Via API
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Hero-Vida's main EV products?",
    "chat_history": []
  }'
```

#### Programmatically

```python
from app.agents.router_agent import route_query

response = route_query(
    query="Compare pricing strategies of top EV companies",
    chat_history=[]
)
print(response['answer'])
```

## API Documentation

### Endpoints

#### Health Check
```
GET /
```
Returns API status and available features.

#### Query Processing
```
POST /query
```
**Request Body:**
```json
{
  "query": "string",
  "chat_history": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ]
}
```

**Response:**
```json
{
  "answer": "string",
  "intent": "string",
  "source_documents": [
    {
      "content": "string",
      "metadata": {
        "filename": "string",
        "page": 1,
        "source_type": "string"
      }
    }
  ],
  "agent_used": "string"
}
```

#### Document Ingestion
```
POST /ingest
```
**Request Body:**
```json
{
  "directory_path": "data"
}
```

#### Multi-Modal Upload
```
POST /upload-multimodal
```
Upload files with multi-modal processing (images, tables, charts).

#### Multi-Modal Query
```
POST /query-multimodal
```
Query with multi-modal content awareness.

## Agent Specifications

### Router Agent

**Purpose**: Intelligent query classification and routing

**Capabilities**:
- Semantic intent detection using sentence embeddings
- Confidence-based agent selection
- Fallback to general QA for ambiguous queries

**Intent Categories**:
- `company`: Company profiles, partnerships, competitive analysis
- `sales`: Sales data, KPIs, market performance
- `pricing`: Pricing strategies, cost analysis

**Configuration**:
```python
INTENT_THRESHOLD = 0.35  # Minimum confidence score
```

### Company Agent

**Purpose**: Company intelligence and competitive analysis

**Specialization**:
- Company profiles and product lines
- Partnership and collaboration analysis
- Market positioning and competitive landscape
- EV/ICE product offerings

**Prompt Engineering**:
- Context-aware responses
- Structured markdown output
- Source attribution
- No hallucination policy

### Sales Agent

**Purpose**: Sales data analysis and KPI tracking

**Specialization**:
- Sales performance metrics
- Regional analysis
- OEM and model-level insights
- Trend identification
- Time-series analysis

**Output Format**:
- Tables for numerical data
- Bullet points for key insights
- Structured headings for clarity

### Pricing Agent

**Purpose**: Pricing intelligence and strategy analysis

**Specialization**:
- Competitive pricing analysis
- Cost structure examination
- Pricing trend identification
- Market segment pricing
- Price-to-feature comparisons

## Multi-Modal Processing

### Supported Content Types

1. **Text Content**
   - Plain text extraction
   - Markdown formatting preservation
   - Metadata extraction

2. **Image Content**
   - Chart and graph analysis
   - Diagram interpretation
   - Visual summary generation
   - OCR for text in images

3. **Table Content**
   - Structure preservation
   - Data extraction
   - Insight generation
   - Formatting detection

### Processing Pipeline

```
Document Input
    │
    ├─► Text Extraction
    │       └─► Chunking & Embedding
    │
    ├─► Image Extraction
    │       └─► Vision API Analysis
    │               └─► Description Generation
    │
    └─► Table Extraction
            └─► Structure Analysis
                    └─► Insight Generation
                            │
                            ▼
                    Combined Multi-Modal Content
                            │
                            ▼
                    Vector Storage (Pinecone)
```

## Project Structure

```
Agentic-RAG-Langchain/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI application entry point
│   │   │
│   │   ├── agents/                      # Specialized agents
│   │   │   ├── __init__.py
│   │   │   ├── router_agent.py          # Query routing and intent detection
│   │   │   ├── company_agent.py         # Company intelligence agent
│   │   │   ├── sales_agent.py           # Sales data analysis agent
│   │   │   ├── pricing_agent.py         # Pricing intelligence agent
│   │   │   └── multimodal_agent.py      # Multi-modal query handler
│   │   │
│   │   ├── api/                         # API routes
│   │   │   ├── __init__.py
│   │   │   └── routes.py                # API endpoint definitions
│   │   │
│   │   ├── core/                        # Core configuration
│   │   │   ├── __init__.py
│   │   │   └── config.py                # Application settings
│   │   │
│   │   ├── services/                    # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py                # Document ingestion service
│   │   │   ├── retriever.py             # Vector retrieval service
│   │   │   ├── stateful_bot.py          # Conversational QA chain
│   │   │   ├── multimodal_processor.py  # Multi-modal processing
│   │   │   ├── vision_service.py        # Image analysis service
│   │   │   └── generator.py             # Response generation
│   │   │
│   │   └── utils/                       # Utility functions
│   │       ├── __init__.py
│   │       ├── logger.py                # Logging configuration
│   │       ├── helpers.py               # Helper functions
│   │       └── image_extractor.py       # Image extraction utilities
│   │
│   ├── client/                          # Streamlit frontend
│   │   ├── app.py                       # Streamlit application
│   │   └── requirements.txt             # Client dependencies
│   │
│   └── requirements.txt                 # Backend dependencies
│
├── data/                                # Document storage (gitignored)
│   └── [Your documents here]
│
├── .gitignore                           # Git ignore rules
├── .env                                 # Environment variables (gitignored)
└── README.md                            # Project documentation
```

## Development Workflow

### Adding New Agents

1. Create agent file in `backend/app/agents/`
2. Implement agent logic with specialized prompts
3. Update router agent intent mapping
4. Add API endpoint if needed
5. Test thoroughly before deployment

### Extending Multi-Modal Capabilities

1. Update `multimodal_processor.py` with new content type
2. Add processing logic in vision service
3. Update chunking strategy if needed
4. Test with sample documents

## Performance Optimization

### Vector Search Optimization

```python
# Adjust retrieval parameters
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Number of results
)
```

### Caching Strategy

Consider implementing caching for:
- Frequently accessed embeddings
- Common query responses
- Preprocessed documents

### Scaling Considerations

- Use Pinecone's serverless tier for auto-scaling
- Implement rate limiting for API endpoints
- Consider batch processing for large ingestion tasks
- Monitor API usage and costs

## Troubleshooting

### Common Issues

**Issue**: Pinecone connection errors
```
Solution: Verify PINECONE_API_KEY and INDEX_NAME in .env file
Check Pinecone dashboard for index status
```

**Issue**: Gemini API quota exceeded
```
Solution: Check API usage in Google AI Studio
Consider rate limiting or upgrade plan
```

**Issue**: Memory errors during ingestion
```
Solution: Process documents in batches
Reduce CHUNK_SIZE if needed
Increase system RAM
```

**Issue**: Slow query responses
```
Solution: Reduce retrieval k parameter
Optimize chunk size
Check Pinecone index performance metrics
```

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Query result caching
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Real-time document monitoring
- [ ] Enhanced image understanding
- [ ] Audio/video content processing
- [ ] Graph-based knowledge representation
- [ ] Fine-tuned domain-specific models
- [ ] Automated testing suite

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- Google AI for Gemini API
- Pinecone for vector database services
- HuggingFace for embedding models
- The open-source community

## Contact

**Developer**: Nitin Singh  
**GitHub**: [@nitinsingh33](https://github.com/nitinsingh33)  
**Repository**: [Agentic-RAG-Langchain](https://github.com/nitinsingh33/Agentic-RAG-Langchain)

---

**Built with** Python, LangChain, FastAPI, and Streamlit
