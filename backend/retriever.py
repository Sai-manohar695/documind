from typing import List, Dict
from embeddings import generate_single_embedding
from vectorstore import search_similar_chunks
import math

def keyword_score(query: str, document: str) -> float:
    """Calculate a simple keyword overlap score between query and document."""
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())
    
    if not query_words or not doc_words:
        return 0.0
    
    # Find common words
    common_words = query_words.intersection(doc_words)
    
    # Remove common stop words from scoring
    stop_words = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "for", 
                  "of", "and", "or", "but", "with", "this", "that", "are", "was"}
    common_words = common_words - stop_words
    
    # Score based on how many query words appear in document
    score = len(common_words) / math.sqrt(len(query_words))
    return score

def hybrid_search(
    query: str,
    n_results: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    collection_name: str = "documind"
) -> List[Dict]:
    """
    Combine semantic search and keyword search for better retrieval.
    semantic_weight + keyword_weight should equal 1.0
    """
    
    # Step 1: Generate embedding for the query
    query_embedding = generate_single_embedding(query)
    
    # Step 2: Get semantic search results from ChromaDB
    # Fetch more than needed so we can re-rank
    raw_results = search_similar_chunks(
        query_embedding=query_embedding,
        n_results=n_results * 2,
        collection_name=collection_name
    )
    
    if not raw_results["documents"][0]:
        return []
    
    # Step 3: Score each result with hybrid scoring
    scored_chunks = []
    
    documents = raw_results["documents"][0]
    distances = raw_results["distances"][0]
    metadatas = raw_results["metadatas"][0]
    
    for doc, distance, metadata in zip(documents, distances, metadatas):
        # Convert distance to similarity score (ChromaDB returns distances)
        semantic_score = 1 - distance
        
        # Calculate keyword score
        kw_score = keyword_score(query, doc)
        
        # Combine scores
        final_score = (semantic_weight * semantic_score) + (keyword_weight * kw_score)
        
        scored_chunks.append({
            "text": doc,
            "semantic_score": round(semantic_score, 4),
            "keyword_score": round(kw_score, 4),
            "final_score": round(final_score, 4),
            "metadata": metadata
        })
    
    # Step 4: Sort by final score and return top n
    scored_chunks.sort(key=lambda x: x["final_score"], reverse=True)
    
    return scored_chunks[:n_results]