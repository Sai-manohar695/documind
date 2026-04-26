import chromadb
from chromadb.config import Settings
from typing import List
import os

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path="chroma_db")

def get_or_create_collection(collection_name: str = "documind"):
    """Get existing collection or create a new one."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def store_embeddings(
    chunks: List[str],
    embeddings: List[List[float]],
    doc_id: str,
    collection_name: str = "documind"
):
    """Store text chunks and their embeddings in ChromaDB."""
    collection = get_or_create_collection(collection_name)
    
    # Create unique IDs for each chunk
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    )
    
    return len(chunks)

def search_similar_chunks(
    query_embedding: List[float],
    n_results: int = 5,
    collection_name: str = "documind"
) -> dict:
    """Search for most similar chunks to a query embedding."""
    collection = get_or_create_collection(collection_name)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    
    return results

def delete_document(doc_id: str, collection_name: str = "documind"):
    """Delete all chunks belonging to a document."""
    collection = get_or_create_collection(collection_name)
    
    # Get all chunk IDs for this document
    results = collection.get(where={"doc_id": doc_id})
    
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return True
    return False

def get_collection_stats(collection_name: str = "documind") -> dict:
    """Get basic stats about the collection."""
    collection = get_or_create_collection(collection_name)
    count = collection.count()
    return {"total_chunks": count, "collection_name": collection_name}