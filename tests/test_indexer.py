import pytest
from pathlib import Path
from src.indexer import get_corpus_files

def test_get_corpus_files_case_insensitivity(tmp_path):
    # Create mock files
    (tmp_path / "valid.pdf").write_text("dummy")
    (tmp_path / "valid.PDF").write_text("dummy")
    (tmp_path / "invalid.jpg").write_text("dummy")
    
    files = get_corpus_files(tmp_path)
    
    assert len(files) == 2
    suffixes = [f.suffix.lower() for f in files]
    assert ".pdf" in suffixes
    assert ".jpg" not in suffixes
