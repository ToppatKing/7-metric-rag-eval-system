import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

@dataclass
class RAGConfig:
    """Centralized configuration for the RAG and Evaluation system."""
    
    # LLM & Embedding Models
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Vector Store Paths
    chroma_persist_dir: str = "./chroma_db"
    
    # Document Processing Parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval Parameters
    top_k: int = 5
    
    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")
