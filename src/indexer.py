import os
import time
import hashlib
import logging
from pathlib import Path
from typing import List
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
            model=config.embedding_model,
            # Built-in retry logic for brief API hiccups
            max_retries=3 
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )

    def build_or_load_index(self, data_dir: str) -> Chroma:
        """Builds or loads a vector store using chunk-batching for massive datasets."""
        folder_hash = hashlib.md5(os.path.abspath(data_dir).encode()).hexdigest()[:8]
        persist_dir = f"{self.config.chroma_persist_dir}_{folder_hash}"
        
        # 1. Initialize the Chroma vector store object
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        
        # 2. Check if the database already contains data
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            # A simple check: if Chroma has documents, assume it's fully built
            if vectorstore._collection.count() > 0:
                print(f"📂 Found existing populated vector cache for '{data_dir}'. Loading...")
                return vectorstore
            
        print(f"⚙️ Building new vector store from '{data_dir}' (Massive Dataset Mode)...")
        
        # 3. Gather all file paths (Lazy Loading setup)
        pdf_files = list(Path(data_dir).rglob("*.pdf"))
        txt_files = list(Path(data_dir).rglob("*.txt"))
        all_files = pdf_files + txt_files
        
        if not all_files:
            raise ValueError(f"No PDF or TXT files found inside '{data_dir}'.")
            
        print(f"📄 Found {len(all_files)} files. Processing in batches to conserve RAM and API limits...")

        # 4. Process files in smaller batches (e.g., 5 files at a time)
        file_batch_size = 5
        
        for i in tqdm(range(0, len(all_files), file_batch_size), desc="Processing File Batches"):
            batch_files = all_files[i:i + file_batch_size]
            batch_docs = []
            
            # Load only the current batch of files into RAM
            for file_path in batch_files:
                try:
                    if file_path.suffix.lower() == '.pdf':
                        batch_docs.extend(PyPDFLoader(str(file_path)).load())
                    else:
                        batch_docs.extend(TextLoader(str(file_path)).load())
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    
            if not batch_docs:
                continue
                
            # Split the current batch into chunks
            chunks = self.text_splitter.split_documents(batch_docs)
            
            # 5. Push to OpenAI/Chroma in micro-batches to respect Embedding Rate Limits
            chunk_batch_size = 100
            for j in range(0, len(chunks), chunk_batch_size):
                micro_batch = chunks[j:j + chunk_batch_size]
                
                # Add documents (this calls the embedding API)
                vectorstore.add_documents(micro_batch)
                
                # Sleep for one second to avoid TPM/RPM rate limits (can be modified)
                time.sleep(1) 
                
        print(f"✅ Vector store successfully built and saved to '{persist_dir}'")
        return vectorstore
