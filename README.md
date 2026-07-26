# DocuMind AI

### RAG-Based Multi-Document Question Answering System

DocuMind AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask questions based on their content.

The system retrieves relevant information from uploaded documents using semantic similarity search and generates context-grounded responses using Llama 3.

## Features

* Multi-PDF document upload
* Semantic document search
* Context-aware question answering
* FAISS vector database
* Google Generative AI embeddings
* Llama 3 via Groq
* Streamlit interface

## Architecture

```text
PDF Documents
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Google Embeddings
      ↓
FAISS Vector Store
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
Llama 3 via Groq
      ↓
Generated Answer
```

## Tech Stack

* **Language:** Python
* **Framework:** LangChain
* **UI:** Streamlit
* **LLM:** Llama 3
* **LLM Inference:** Groq
* **Embeddings:** Google Generative AI
* **Vector Database:** FAISS
* **PDF Processing:** PyPDFLoader

## Installation

```bash
git clone https://github.com/your-username/DocuMind-AI.git
cd DocuMind-AI
pip install -r requirements.txt
```

Create a `.env` file:

```env
API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Run the application:

```bash
streamlit run app.py
```

## Workflow

1. Upload one or more PDF documents.
2. Process the documents to create the FAISS vector database.
3. Ask questions about the uploaded documents.
4. Receive answers generated from the retrieved document context.

## Future Enhancements

* Source citations with page numbers
* OCR support for scanned PDFs
* Hybrid search
* Reranking
* Conversation memory
* Document comparison

## Author

**Madhuraa**
B.Tech Computer Engineering Student
