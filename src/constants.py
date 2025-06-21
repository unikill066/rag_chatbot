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


SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader,
    }