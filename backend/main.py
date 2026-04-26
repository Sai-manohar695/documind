import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion import process_pdf
from embeddings import generate_embeddings
from vectorstore import store_embeddings, delete_document, get_collection_stats
from qa_chain import ask_question

load_dotenv()

app = FastAPI(
    title="DocuMind API",
    description="Intelligent Document Q&A Engine",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------- Request Models -----------

class QuestionRequest(BaseModel):
    query: str
    doc_id: str


class DeleteRequest(BaseModel):
    doc_id: str


# ----------- Routes -----------

@app.get("/")
def root():
    return {"message": "DocuMind API is running!"}


@app.get("/stats")
def get_stats():
    """Get collection statistics."""
    stats = get_collection_stats()
    return stats


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, process it, generate embeddings,
    and store in ChromaDB.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )
    
    # Save file to disk
    doc_id = file.filename.replace(".pdf", "").replace(" ", "_").lower()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF into chunks
    chunks = process_pdf(file_path, chunk_size=512)
    
    # Generate embeddings for all chunks
    embeddings = generate_embeddings(chunks)
    
    # Store in ChromaDB
    num_chunks = store_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        doc_id=doc_id
    )
    
    return {
        "message": "PDF uploaded and processed successfully.",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": num_chunks
    }


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Ask a question against an uploaded document.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )
    
    result = ask_question(
        query=request.query,
        collection_name="documind"
    )
    
    return {
        "query": result["query"],
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": [
            {
                "text": chunk["text"][:200] + "...",
                "score": chunk["final_score"]
            }
            for chunk in result["chunks_used"]
        ]
    }


@app.delete("/document")
async def delete_doc(request: DeleteRequest):
    """
    Delete a document and all its chunks from ChromaDB.
    """
    success = delete_document(doc_id=request.doc_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found."
        )
    
    return {"message": f"Document '{request.doc_id}' deleted successfully."}