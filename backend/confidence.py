from typing import List, Dict

def calculate_confidence(retrieved_chunks: List[Dict]) -> Dict:
    """
    Calculate confidence score based on retrieval quality.
    Returns a confidence level and reasoning.
    """
    
    if not retrieved_chunks:
        return {
            "score": 0.0,
            "level": "none",
            "message": "No relevant chunks found in the document."
        }
    
    # Get the top chunk's final score
    top_score = retrieved_chunks[0]["final_score"]
    
    # Get average score across all retrieved chunks
    avg_score = sum(chunk["final_score"] for chunk in retrieved_chunks) / len(retrieved_chunks)
    
    # Check score consistency — are all chunks similarly relevant?
    scores = [chunk["final_score"] for chunk in retrieved_chunks]
    score_spread = max(scores) - min(scores)
    
    # Calculate final confidence
    confidence_score = (top_score * 0.6) + (avg_score * 0.4)
    
    # Penalize if scores are very inconsistent
    if score_spread > 0.5:
        confidence_score *= 0.85
    
    confidence_score = round(confidence_score, 4)
    
    # Assign confidence level
    if confidence_score >= 0.75:
        level = "high"
        message = "Answer is strongly supported by the document."
    elif confidence_score >= 0.50:
        level = "medium"
        message = "Answer is moderately supported. Verify with the document."
    elif confidence_score >= 0.30:
        level = "low"
        message = "Answer may not be fully accurate. Limited support found."
    else:
        level = "very_low"
        message = "Answer is likely unreliable. Relevant content not found."
    
    return {
        "score": confidence_score,
        "level": level,
        "message": message,
        "top_chunk_score": round(top_score, 4),
        "avg_chunk_score": round(avg_score, 4)
    }


def should_answer(confidence: Dict, threshold: float = 0.30) -> bool:
    """
    Decide whether to answer or flag the response as unreliable.
    Returns False if confidence is too low.
    """
    return confidence["score"] >= threshold