import pytest
from pathlib import Path
from src.indexer import get_corpus_files

def test_get_corpus_files_case_insensitivity(tmp_path):
    # Setup mock files in a temporary directory
    (tmp_path / "doc1.txt").write_text("Hello")
    (tmp_path / "doc2.PDF").write_text("World")
    (tmp_path / "doc3.Md").write_text("Markdown")
    (tmp_path / "image.png").write_text("ignore me")
    
    # Run discovery
    files = get_corpus_files(tmp_path)
    
    # Verify exact supported file discovery
    assert len(files) == 3
    file_names = [f.name for f in files]
    assert "doc1.txt" in file_names
    assert "doc2.PDF" in file_names
    assert "doc3.Md" in file_names
    assert "image.png" not in file_names
