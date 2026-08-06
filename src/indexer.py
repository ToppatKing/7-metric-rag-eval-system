import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# Supported document extensions (case-insensitive)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def get_corpus_files(directory_path: str | Path) -> List[Path]:
    """
    Recursively discover all supported corpus files in a directory,
    matching file extensions case-insensitively.
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        logger.warning(f"Corpus directory does not exist or is not a directory: {directory}")
        return []

    discovered_files = [
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(discovered_files)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a single file's content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(
    files: List[Path],
    corpus_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str
) -> Dict[str, Any]:
    """
    Generate an index manifest payload containing content hashes, file paths, 
    and indexing configuration to guarantee cache validity.
    """
    file_manifest: Dict[str, str] = {}
    combined_hash = hashlib.sha256()

    for file_path in files:
        rel_path = str(file_path.relative_to(corpus_dir))
        file_hash = compute_file_hash(file_path)
        file_manifest[rel_path] = file_hash
        combined_hash.update(f"{rel_path}:{file_hash}".encode("utf-8"))

    # Include key configuration options in the aggregate hash signature
    config_signature = f"{embedding_model_name}:{chunk_size}:{chunk_overlap}"
    combined_hash.update(config_signature.encode("utf-8"))

    return {
        "manifest_version": "1.0.0",
        "aggregate_hash": combined_hash.hexdigest(),
        "embedding_model": embedding_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_files_discovered": len(files),
        "files": file_manifest,
    }


class Indexer:
    """
    Document Indexer featuring manifest-backed persistence, atomic re-indexing,
    case-insensitive file discovery, and comprehensive indexing metrics.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_function: Any,
        embedding_model_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "rag_collection",
    ):
        self.persist_directory = Path(persist_directory)
        self.embedding_function = embedding_function
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.manifest_path = self.persist_directory / "manifest.json"

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _is_cache_valid(self, current_manifest: Dict[str, Any]) -> bool:
        """Verify if the existing on-disk index matches the current corpus and parameters."""
        if not self.persist_directory.exists() or not self.manifest_path.exists():
            return False

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)

            return (
                cached_manifest.get("aggregate_hash") == current_manifest["aggregate_hash"]
                and cached_manifest.get("embedding_model") == self.embedding_model_name
                and cached_manifest.get("chunk_size") == self.chunk_size
                and cached_manifest.get("chunk_overlap") == self.chunk_overlap
            )
        except Exception as e:
            logger.warning(f"Failed to read or validate index manifest: {e}")
            return False

    def _clear_cache(self) -> None:
        """Safely purge the existing index directory to ensure a clean build."""
        if self.persist_directory.exists():
            logger.info(f"Purging existing index directory at: {self.persist_directory}")
            shutil.rmtree(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    def _load_document(self, file_path: Path) -> List[Document]:
        """Load document depending on file extension."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        elif ext in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
            return loader.load()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def build_or_load_index(
        self,
        corpus_directory: str | Path,
        force_reindex: bool = False
    ) -> Chroma:
        """
        Loads an existing valid vector database index or builds a new one transactionally.
        """
        corpus_dir = Path(corpus_directory)
        discovered_files = get_corpus_files(corpus_dir)

        if not discovered_files:
            raise ValueError(f"No valid corpus documents (.pdf, .txt, .md) found in {corpus_dir}")

        current_manifest = generate_manifest(
            files=discovered_files,
            corpus_dir=corpus_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model_name=self.embedding_model_name,
        )

        # 1. Reuse existing index if manifest matches and no force re-index requested
        if not force_reindex and self._is_cache_valid(current_manifest):
            logger.info("Valid cached index and manifest confirmed. Loading existing collection...")
            return Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )

        if force_reindex:
            logger.info("Force re-index flag detected.")
        else:
            logger.info("Index cache is stale, incomplete, or missing. Triggering full build...")

        # 2. Reset database directory for fresh build
        self._clear_cache()

        # 3. Process documents and capture processing diagnostics
        loaded_count = 0
        failed_count = 0
        empty_count = 0
        all_chunks: List[Document] = []

        for file_path in discovered_files:
            try:
                docs = self._load_document(file_path)
                
                # Exclude empty pages or corrupt blank files
                valid_docs = [d for d in docs if d.page_content and d.page_content.strip()]
                
                if not valid_docs:
                    logger.warning(f"File loaded but contained no readable text: {file_path}")
                    empty_count += 1
                    continue

                chunks = self.text_splitter.split_documents(valid_docs)
                all_chunks.extend(chunks)
                loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to process source file '{file_path}': {e}")
                failed_count += 1

        if not all_chunks:
            raise RuntimeError("Indexing aborted: Zero valid text chunks were extracted from the corpus.")

        logger.info(
            f"Indexing Diagnostics: Discovered={len(discovered_files)} | "
            f"Loaded={loaded_count} | Empty={empty_count} | "
            f"Failed={failed_count} | Total Chunks={len(all_chunks)}"
        )

        # 4. Build store using public Chroma API
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embedding_function,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
        )

        # 5. Write completion marker manifest only upon complete success
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(current_manifest, f, indent=2)

        logger.info(f"Index successfully built and promoted. Manifest written to {self.manifest_path}")
        return vectorstore
