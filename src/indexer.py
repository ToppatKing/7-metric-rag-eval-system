import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            api_key=config.openai_api_key, 
            model=config.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )

    def load_documents(self, data_dir: str) -> List:
        """Loads all PDF and TXT files from the specified data directory."""
        logger.info(f"Loading documents from {data_dir}...")
        
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        
        docs = pdf_loader.load() + txt_loader.load()
        logger.info(f"Successfully loaded {len(docs)} documents.")
        return docs

    def build_or_load_index(self, data_dir: str = "data/") -> Chroma:
        """Builds a new ChromaDB index from documents or loads an existing one."""
        import os
        
        # If DB already exists locally, just load it to save time and API costs
        if os.path.exists(self.config.chroma_persist_dir):
            logger.info("Found existing ChromaDB. Loading from disk...")
            return Chroma(
                persist_directory=self.config.chroma_persist_dir,
                embedding_function=self.embeddings
            )
            
        docs = self.load_documents(data_dir)
        if not docs:
            raise ValueError("No documents found in the data directory!")
            
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(chunks)} contextual chunks.")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config.chroma_persist_dir
        )
        logger.info(f"Vector store successfully persisted to {self.config.chroma_persist_dir}")
        return vectorstore
