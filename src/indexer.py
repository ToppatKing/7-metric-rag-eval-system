import os
import hashlib
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import RAGConfig

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
        """Loads all PDF and TXT files from the user-specified directory."""
        logger.info(f"Loading documents from: {data_dir}")
        
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        
        docs = pdf_loader.load() + txt_loader.load()
        logger.info(f"Successfully loaded {len(docs)} document chunks/pages.")
        return docs

    def build_or_load_index(self, data_dir: str) -> Chroma:
        """Builds or loads a vector store unique to the provided data_dir."""
        # Create a unique database directory hash based on the folder path
        folder_hash = hashlib.md5(os.path.abspath(data_dir).encode()).hexdigest()[:8]
        persist_dir = f"{self.config.chroma_persist_dir}_{folder_hash}"
        
        # If an index already exists for this exact directory, load it
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print(f"📂 Found existing vector cache for '{data_dir}'. Loading...")
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            
        print(f"⚙️ Building new vector store from '{data_dir}'...")
        docs = self.load_documents(data_dir)
        if not docs:
            raise ValueError(f"No PDF or TXT files found inside '{data_dir}'.")
            
        chunks = self.text_splitter.split_documents(docs)
        print(f"✂️ Split documents into {len(chunks)} contextual chunks.")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        print(f"✅ Vector store successfully saved to '{persist_dir}'")
        return vectorstore
