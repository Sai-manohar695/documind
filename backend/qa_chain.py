import os
from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv
from retriever import hybrid_search
from confidence import calculate_confidence, should_answer

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(query: str, chunks: List[Dict]) -> str:
    """Build a prompt with retrieved context for the LLM."""
    
    # Combine retrieved chunks into context
    context = "\n\n".join([
        f"[Chunk {i+1}]:\n{chunk['text']}" 
        for i, chunk in enumerate(chunks)
    ])
    
    prompt = f"""You are a helpful document assistant. Answer the user's question based ONLY on the provided context below.

If the answer is not found in the context, say "I could not find this information in the document." 
Do NOT make up information or use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt


def ask_question(query: str, collection_name: str = "documind") -> Dict:
    """
    Full Q&A pipeline:
    1. Retrieve relevant chunks
    2. Check confidence
    3. Call LLM if confidence is sufficient
    4. Return answer with metadata
    """
    
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = hybrid_search(
        query=query,
        n_results=5,
        collection_name=collection_name
    )
    
    # Step 2: Calculate confidence
    confidence = calculate_confidence(retrieved_chunks)
    
    # Step 3: Check if we should answer
    if not should_answer(confidence):
        return {
            "answer": "I could not find relevant information in the document to answer this question.",
            "confidence": confidence,
            "chunks_used": [],
            "query": query
        }
    
    # Step 4: Build prompt with context
    prompt = build_prompt(query, retrieved_chunks)
    
    # Step 5: Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a precise document assistant. Only answer based on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )
    
    answer = response.choices[0].message.content.strip()
    
    return {
        "answer": answer,
        "confidence": confidence,
        "chunks_used": retrieved_chunks,
        "query": query
    }