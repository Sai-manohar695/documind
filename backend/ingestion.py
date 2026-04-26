import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

def process_pdf(file_path: str, chunk_size: int = 512) -> list[str]:
    """Full pipeline: load PDF and return chunks."""
    text = load_pdf(file_path)
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF.")
    chunks = chunk_text(text, chunk_size=chunk_size)
    return chunks