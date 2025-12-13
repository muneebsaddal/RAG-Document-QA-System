# RAG Document Q&A System

A complete Retrieval-Augmented Generation (RAG) system that allows users to upload documents (PDF or TXT) and ask questions based on their content using Mistral or other models via Ollama.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Future Enhancements](#future-enhancements)

## Overview

This project implements a production-ready RAG (Retrieval-Augmented Generation) pipeline - the foundation of most enterprise GenAI applications. It demonstrates the complete workflow from document ingestion to question answering, with built-in hallucination prevention.

## Features

- **Document Upload**: Support for PDF and TXT files via drag-and-drop or file selection
- **Intelligent Chunking**: Automatic text splitting with overlap for better context preservation
- **TF-IDF Retrieval**: Fast and reliable text-based similarity search without embeddings
- **Context-Aware Responses**: LLM generates answers based only on document content
- **Hallucination Prevention**: Refuses to answer questions not covered in the document
- **Real-time Chat Interface**: Interactive web-based UI with context visualization
- **REST API**: Clean Flask API for integration with other systems
- **Lightweight**: Minimal dependencies, runs entirely locally

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (HTML/CSS/JS)                  │
│  - Document Upload                                           │
│  - Chat Interface                                            │
│  - Context Display                                           │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────────┐
│                    Flask API Server (app.py)                 │
│  - /api/upload   - /api/query   - /api/reset                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                RAG System (rag_system.py)                    │
│                                                              │
│  1. Document Ingestion                                       │
│     └─> Load PDF/TXT → Split into chunks                    │
│                                                              │
│  2. Retrieval (TF-IDF)                                       │
│     └─> Query → Tokenize → Score chunks → Top-K             │
│                                                              │
│  3. Generation                                               │
│     └─> Context + Query → Ollama → Answer                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Ollama (Local LLM)                          │
│  - Model: gemma3:270m / mistral / custom                     │
│  - Endpoint: localhost:11434                                 │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running
- At least one LLM model pulled in Ollama (e.g., gemma3:270m, mistral)
- 4GB RAM minimum (8GB recommended)

## Installation

### Step 1: Install Ollama

**Windows:**
Download from https://ollama.com/download

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Pull a Language Model

```bash
# Lightweight model (recommended for testing)
ollama pull gemma3:270m

# Or a more capable model
ollama pull mistral
```

Verify installation:
```bash
ollama list
```

### Step 3: Clone/Download Project

```bash
mkdir rag-document-qa
cd rag-document-qa
```

Place the following files in the directory:
- `rag_system.py` - Core RAG logic
- `app.py` - Flask API server
- `index.html` - Frontend interface
- `test_ollama.py` - Connection testing utility
- `requirements.txt` - Python dependencies

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install chromadb==0.4.22 pypdf==3.0.0 flask==3.0.0 flask-cors==4.0.0 requests==2.31.0
```

### Step 5: Create Upload Directory

```bash
mkdir uploads
```

## Usage

### Starting the System

**Terminal 1: Start Ollama Server**

For CPU mode (recommended if you encounter GPU errors):
```bash
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=""
ollama serve

# Linux/Mac
CUDA_VISIBLE_DEVICES="" ollama serve
```

For GPU mode:
```bash
ollama serve
```

**Terminal 2: Start Flask Backend**
```bash
python app.py
```

Expected output:
```
Starting RAG Server...
Server running at: http://localhost:5000
```

**Terminal 3: Serve Frontend**
```bash
python -m http.server 8000
```

Then open your browser to: `http://localhost:8000/index.html`

### Using the Interface

1. **Upload Document**: Click or drag-and-drop a PDF/TXT file
2. **Wait for Processing**: System will chunk and index the document
3. **Ask Questions**: Type questions in the chat input
4. **View Context**: Click context chips to see which chunks were used

## Project Structure

```
rag-document-qa/
├── rag_system.py         # Core RAG implementation
├── app.py                # Flask REST API
├── index.html            # Web interface
├── test_ollama.py        # Testing utility
├── requirements.txt      # Python dependencies
├── uploads/              # Uploaded documents (created at runtime)
└── README.md            # This file
```

## How It Works

### 1. Document Ingestion

```python
Document (PDF/TXT) 
    → Extract text
    → Split into ~500 character chunks with 50 char overlap
    → Store chunks in memory
```

### 2. Query Processing

```python
User Question
    → Tokenize query (lowercase, remove punctuation)
    → Compute TF-IDF scores against all chunks
    → Retrieve top 3 most relevant chunks
```

### 3. Answer Generation

```python
Retrieved Chunks + User Question
    → Construct RAG prompt with context
    → Send to Ollama LLM
    → Return generated answer
```

### TF-IDF Algorithm

The system uses Term Frequency-Inverse Document Frequency for retrieval:

- **Term Frequency (TF)**: How often a term appears in a document
- **Inverse Document Frequency (IDF)**: How unique/rare a term is across all documents
- **Score**: TF × IDF for each query term, summed across the document

This approach is:
- Fast (no embedding generation)
- Reliable (deterministic results)
- Lightweight (no ML models for retrieval)
- Effective for keyword-based search

## Testing

### Automated Testing

Run the test script to verify Ollama connectivity:
```bash
python test_ollama.py
```

This checks:
1. Ollama server is running
2. Models are available
3. Text generation works
4. RAG-style prompts work

### Manual Testing

#### Test 1: Basic Information Retrieval
- Upload a document (e.g., your CV, technical manual)
- Ask: "What is this document about?"
- Expected: Summary based on document content

#### Test 2: Specific Information
- Ask: "What are the key points mentioned?"
- Expected: Specific details extracted from the document

#### Test 3: Hallucination Prevention (Critical)
- Ask: "What is the weather today?"
- Expected: "I cannot find this information in the provided document."

#### Test 4: Context Viewing
- Click on "Context 1", "Context 2", "Context 3" chips
- Verify the displayed chunks are relevant to your question

## Troubleshooting

### Issue: "Cannot connect to Ollama"
**Solution**: Make sure Ollama is running
```bash
ollama serve
```

### Issue: "CUDA error" or "out of memory"
**Solution**: Run Ollama in CPU mode
```bash
CUDA_VISIBLE_DEVICES="" ollama serve
```

### Issue: Slow responses
**Solution**: 
- Use a smaller model (gemma3:270m instead of mistral)
- Ensure Ollama is in CPU mode if GPU is causing issues
- Local models typically take 10-30 seconds per query

### Issue: "No documents have been indexed"
**Solution**: Upload a document before asking questions

### Issue: CORS errors in browser
**Solution**: 
- Make sure Flask server is running with CORS enabled (already configured)
- Serve frontend through http.server, not file:// protocol

### Issue: Poor answer quality
**Solutions**:
- Use a more capable model (mistral instead of gemma3:270m)
- Ensure your question relates to document content
- Try rephrasing your question with keywords from the document

## API Documentation

### POST /api/upload
Upload and ingest a document.

**Request:**
- Content-Type: multipart/form-data
- Body: file (PDF or TXT)

**Response:**
```json
{
  "success": true,
  "message": "Document 'example.pdf' ingested successfully",
  "filename": "example.pdf"
}
```

### POST /api/query
Query the RAG system.

**Request:**
```json
{
  "question": "What is this document about?"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "This document discusses...",
  "context": [
    "Chunk 1 text...",
    "Chunk 2 text...",
    "Chunk 3 text..."
  ]
}
```

### POST /api/reset
Clear all documents and reset the system.

**Response:**
```json
{
  "success": true,
  "message": "System reset successfully"
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "RAG server is running"
}
```

## Technical Details

### Dependencies

- **chromadb**: Vector database (used for storage, not embeddings)
- **pypdf**: PDF text extraction
- **flask**: Web server and REST API
- **flask-cors**: Cross-origin resource sharing
- **requests**: HTTP client for Ollama API

### Configuration

**Model Selection** (in app.py):
```python
rag = RAGSystem(model_name="gemma3:270m")
```

**Chunk Size** (in rag_system.py):
```python
chunk_size = 500  # characters
overlap = 50      # character overlap between chunks
```

**Retrieval Parameters** (in rag_system.py):
```python
top_k = 3  # number of chunks to retrieve
```

**Generation Parameters** (in rag_system.py):
```python
"options": {
    "temperature": 0.7,
    "num_predict": 500
}
```

### Performance Considerations

- **Document Size**: Works best with documents under 100 pages
- **Chunk Size**: 500 characters balances context and precision
- **Top-K**: 3 chunks provides sufficient context without overwhelming the LLM
- **Response Time**: 10-30 seconds per query depending on model and hardware

## Future Enhancements

### Potential Improvements

1. **Multi-document Support**: Handle multiple uploaded documents simultaneously
2. **Better Chunking**: Implement semantic chunking (paragraph/section-based)
3. **Chat History**: Add conversation memory for follow-up questions
4. **Source Citations**: Include page numbers and exact locations in responses
5. **Streaming Responses**: Stream LLM output for better user experience
6. **Document Management**: UI for viewing and deleting uploaded documents
7. **Advanced Retrieval**: Implement hybrid search (TF-IDF + semantic embeddings)
8. **Export Functionality**: Download chat history or extracted information
9. **Authentication**: Add user accounts and document privacy
10. **Analytics**: Track common questions and system usage

### Advanced Features

- **Query Rewriting**: Automatically improve user questions
- **Answer Validation**: Cross-reference multiple chunks for accuracy
- **Custom Models**: Support for fine-tuned domain-specific models
- **Batch Processing**: Process multiple documents at once
- **API Key Management**: Secure multi-user deployments

## Learning Outcomes

This project demonstrates the complete RAG pipeline used in 90% of enterprise GenAI applications:

1. **Document Processing**: Text extraction and intelligent chunking
2. **Information Retrieval**: TF-IDF algorithm for similarity search
3. **Prompt Engineering**: Constructing effective RAG prompts
4. **Hallucination Prevention**: Constraining LLM responses to source material
5. **Full-Stack Integration**: Backend API + Frontend UI + LLM orchestration
6. **Production Considerations**: Error handling, testing, and deployment

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify Ollama is running: `ollama list`
3. Test connectivity: `python test_ollama.py`
4. Review Flask logs for error messages

## Acknowledgments

Built with:
- Ollama for local LLM inference
- Flask for web framework
- ChromaDB for document storage
- Classic information retrieval algorithms (TF-IDF)