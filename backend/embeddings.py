from sentence_transformers import SentenceTransformer
from typing import List

# Load model once when the module is imported
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """Convert text chunks into vector embeddings."""
    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )
    # Convert numpy arrays to plain Python lists
    return [embedding.tolist() for embedding in embeddings]

def generate_single_embedding(text: str) -> List[float]:
    """Convert a single piece of text into a vector embedding."""
    embedding = model.encode(text)
    # Convert numpy array to plain Python list
    return embedding.tolist()