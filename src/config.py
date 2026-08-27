import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env or Codespace Secrets
load_dotenv()

@dataclass
class RAGConfig:
    """Centralized configuration for the RAG and Evaluation system."""
    
    # Credentials
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # Models
    llm_model: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
    embedding_model: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    # Document Processing Parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Execution Parameters
    top_k: int = 64
    num_trials: int = 3
    
    def __post_init__(self):
        # Now it checks if EITHER standard OpenAI OR Azure keys are present
        if not self.openai_api_key and not (self.azure_api_key and self.azure_endpoint):
            raise ValueError("API credentials (Azure or standard OpenAI) are missing. Please check your .env file or Codespace Secrets.")
