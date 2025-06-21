import os, tempfile
import pytest
from src.utils.document_loader import DocumentLoader, DocumentLoadError
from src.constants import SUPPORTED_EXTENSIONS

@pytest.fixture
def loader():
    return DocumentLoader(SUPPORTED_EXTENSIONS)

def make_temp_file(tmp_path, name, content: bytes):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)

def test_load_txt(loader, tmp_path):
    path = make_temp_file(tmp_path, "./docs/foo.txt", b"just a test")
    docs = loader.load_document(path)
    assert len(docs) == 1
    assert "just a test" in docs[0].page_content

def test_load_md(loader, tmp_path):
    path = make_temp_file(tmp_path, "./docs/bar.md", b"# Header\ncontent")
    docs = loader.load_documents([path])
    assert any("Header" in d.page_content for d in docs)

def test_unsupported_ext(loader, tmp_path):
    path = make_temp_file(tmp_path, "./docs/bad.exe", b"binary")
    with pytest.raises(ValueError):
        loader.load_document(path)

def test_load_directory(loader, tmp_path):
    f1 = make_temp_file(tmp_path, "./docs/a.txt", b"a")
    f2 = make_temp_file(tmp_path, "./docs/b.txt", b"b")
    docs = loader.load_directory(str(tmp_path))
    assert len(docs) == 2

def test_stats_empty(loader):
    stats = loader.get_document_stats([])
    assert stats["total_documents"] == 0

def test_stats(loader, tmp_path):
    path = make_temp_file(tmp_path, "./docs/c.txt", b"xyz")
    docs = loader.load_document(path)
    stats = loader.get_document_stats(docs)
    assert stats["total_documents"] == 1
    assert stats["total_characters"] == 3