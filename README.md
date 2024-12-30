# RAG-Document-QandA-with-NVIDIA-NIM (Llama3 RAG Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot powered by Llama3, NVIDIA NIM, and HuggingFace embeddings for document-based question answering.

## Features

- PDF document processing and embedding
- Vector similarity search using FAISS
- Document chunking and retrieval
- Real-time response generation
- Response time tracking
- Document similarity exploration

## Installation

```bash
pip install -r requirements.txt
```

## Required Dependencies

- streamlit
- langchain-nvidia-ai-endpoints
- langchain-community
- langchain-core
- langchain-huggingface
- faiss-cpu
- python-dotenv

## Environment Setup

Create a `.env` file with the following:

```plaintext
NVIDIA_API_KEY=your_nvidia_api_key
HF_TOKEN=your_huggingface_token
```

## Project Structure

```plaintext
.
├── app.py
├── pdfs_folder/
│   └── (your PDF documents)
├── .env
└── requirements.txt
```

## Configuration

**Model Settings**
- LLM: meta/llama3-70b-instruct
- Embeddings: all-MiniLM-L6-v2
- Chunk size: 1500
- Chunk overlap: 100

## Usage

1. Place PDF documents in the `pdfs_folder` directory
2. Start the application:
```bash
streamlit run app.py
```
3. Click "Document Embedding" to process documents
4. Enter your question about the documents
5. View responses and similar document chunks

## Features Breakdown

**Document Processing**
- PDF directory loading
- Recursive text splitting
- Vector embeddings generation
- FAISS vector store creation

**Query Processing**
- Context-aware prompting
- Document retrieval
- Response generation
- Performance timing

Sample Output:
<img width="1470" alt="sample output" src="https://github.com/user-attachments/assets/c48244cf-9578-4dc8-b271-88584cf11bd7" />
