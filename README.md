# 📄 DocuMind — Intelligent Document Q&A Engine

An end-to-end RAG (Retrieval-Augmented Generation) pipeline that lets users upload PDFs and ask natural language questions against them.

## 🚀 Live Demo
[DocuMind on Streamlit Cloud](your-streamlit-url-here)

## 🧠 How It Works

1. **Upload** a PDF document
2. **Ingestion** — text is extracted, chunked into 512-token pieces
3. **Embedding** — chunks are converted to vectors using `all-MiniLM-L6-v2`
4. **Storage** — vectors stored in ChromaDB for fast retrieval
5. **Query** — user asks a question, hybrid search finds relevant chunks
6. **Answer** — Groq LLM (Llama 3.1) generates a grounded answer with confidence scoring

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | Groq API (llama-3.1-8b-instant) |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Processing | PyPDF2 |
| Text Splitting | LangChain Text Splitters |

## ✨ Key Features

- **Hybrid Search** — combines semantic (70%) and keyword (30%) search for better retrieval
- **Confidence Scoring** — flags low-certainty answers to reduce hallucinations
- **Source Attribution** — shows which document chunks were used to generate the answer
- **Document Management** — upload, query, and delete documents

## 📊 Design Decisions

### Chunk Size
Used 512 tokens with 50 token overlap after experimenting with 256 and 1024:
- 256 tokens — too granular, lost context
- 512 tokens — best balance of context and precision
- 1024 tokens — too broad, reduced retrieval accuracy

### Embedding Model
Chose `all-MiniLM-L6-v2` over OpenAI embeddings:
- Runs completely locally — zero cost
- 384-dimensional vectors — fast and memory efficient
- Strong performance on semantic similarity tasks

### Hybrid Search
Pure semantic search misses exact keyword matches. Combined with keyword scoring:
- Semantic search handles conceptual similarity
- Keyword search handles specific term matching
- 70/30 weighting after testing multiple ratios

## 🏃 Running Locally

### Prerequisites
- Python 3.11+
- Groq API key (free at console.groq.com)

### Setup

1. Clone the repository
```bash
git clone https://github.com/Sai-manohar695/documind.git
cd documind
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your Groq API key to `.env`
GROQ_API_KEY=your_key_here

5. Start the backend
```bash
cd backend
uvicorn main:app --reload
```

6. Start the frontend (new terminal)
```bash
cd frontend
streamlit run app.py
```

7. Open http://localhost:8501

## 📁 Project Structure
documind/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── ingestion.py      # PDF processing & chunking
│   ├── embeddings.py     # Sentence transformer embeddings
│   ├── vectorstore.py    # ChromaDB operations
│   ├── retriever.py      # Hybrid search
│   ├── qa_chain.py       # LLM Q&A pipeline
│   └── confidence.py     # Confidence scoring
├── frontend/
│   └── app.py            # Streamlit UI
├── evaluation/
│   └── evaluate.py       # Evaluation pipeline
└── requirements.txt

## 🔮 Future Improvements

- RAGAS evaluation integration for deeper metrics
- Support for multiple documents simultaneously
- Domain-specific embedding models for technical papers
- Conversation memory across multiple questions