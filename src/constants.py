"""
"""

# imports
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, JSONLoader, PyPDFLoader, Docx2txtLoader, UnstructuredXMLLoader, UnstructuredMarkdownLoader


SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': TextLoader,
        '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader
    }


    