"""
"""

# imports
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    JSONLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredXMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
from pathlib import Path


SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader,
    }

VECTOR_DB_FP = Path(__file__).parent.parent / "faiss_vector_store"