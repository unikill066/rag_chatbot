"""
Author: Nikhil Nageshwar Inturi (GitHub: unikill066, email: inturinikhilnageshwar@gmail.com)

Document Processor Module
Handles reading various document formats: .docx, .pdf, .html, .xml, .md, .txt
"""

# imports
import os, logging
from pathlib import Path
from typing import List, Optional, Any, Type, Dict
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, JSONLoader, PyPDFLoader, Docx2txtLoader, UnstructuredXMLLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document
# from constants import SUPPORTED_EXTENSIONS
SUPPORTED_EXTENSIONS = {'.pdf': PyPDFLoader, '.txt': TextLoader, '.md': UnstructuredMarkdownLoader, '.csv': CSVLoader, '.docx': Docx2txtLoader,}

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when a document fails to load."""
    pass

class DocumentLoader:
    """DocumentLoader class - Loads .docx, .pdf, .html, .xml, .md, .txt, .csv and .xlsx documents."""
    
    def __init__(self, supported_extensions: Dict[str, Type]):
        self.supported_extensions = supported_extensions
        self.processed_files: List[str]= list()

    def load_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load a single document based on its file extension
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata to attach to the document
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        try:
            loader_class = self.supported_extensions[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()

            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            for doc in documents:
                doc.metadata.update({
                    'source_file': str(file_path),
                    'file_type': extension,
                    'file_size': file_path.stat().st_size,
                    'file_name': file_path.name
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            self.processed_files.append(str(file_path))
            return documents
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise DocumentLoadError(f"Failed to load {file_path}") from e
        
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents based on the file extension(s)
        
        Args:
            file_path: Path to the document(s)
            metadata: Additional metadata to attach to the document(s)
            
        Returns:
            List of Document objects
        """
        documents = list()
        if not isinstance(file_paths, list):
            raise TypeError(f"Expected a list, but got {file_paths}")
        for file_path in file_paths:
            try:
                document = self.load_document(file_path=file_path)
                documents.extend(document)
            except Exception as e:
                logger.warning(f"Skipping {file_path} due to error: {e}")
                continue  # processes next document

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def load_directory(self, directory_path: str, recursive: bool = True, file_pattern: str = "*", metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            file_pattern: Pattern to match files (e.g., "*.pdf")
            metadata: Additional metadata to attach to all documents
            
        Returns:
            List of all loaded Document objects
        """
        directory_path, documents = Path(directory_path), list()

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if recursive:
            files = directory_path.rglob(file_pattern)
        else: 
            files = directory_path.glob(file_pattern)

        for file in files:
            if file.is_file() and file.suffix.lower() in self.supported_extensions:
                try:
                    document = self.load_document(file_path=str(file), metadata=metadata)
                    documents.extend(document)
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {str(e)}")
                    continue 

        logger.info(f"Loaded {len(documents)} total documents from {directory_path}")
        return documents

    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        file_types, total_chars, sources = {}, 0, set()
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_chars += len(doc.page_content)
            source = doc.metadata.get('source_file', 'unknown')
            sources.add(source)
        return {"total_documents": len(documents), "total_characters": total_chars,
            "average_chars_per_doc": total_chars // len(documents), "unique_sources": len(sources),
            "file_types": file_types, "processed_files": self.processed_files}

# # testing
# SUPPORTED_EXTENSIONS = {'.pdf': PyPDFLoader, '.txt': TextLoader, '.md': UnstructuredMarkdownLoader, '.csv': CSVLoader, '.docx': Docx2txtLoader,}
# doc_loader = DocumentLoader(SUPPORTED_EXTENSIONS)
# print(len(doc_loader.load_document("/Users/discovery/Desktop/rag_chatbot/docs/Nikhil_Nageshwar_InturiG.pdf")[0].page_content))
# print(doc_loader.load_documents(["/Users/discovery/Desktop/rag_chatbot/docs/Nikhil_Nageshwar_InturiG.pdf", "/Users/discovery/Desktop/rag_chatbot/docs/Nikhil_Nageshwar_Inturi_cr.pdf"]))
# print(doc_loader.load_directory("/Users/discovery/Desktop/rag_chatbot/docs"))
# print(doc_loader.get_document_stats(doc_loader.load_documents(["/Users/discovery/Desktop/rag_chatbot/docs/Nikhil_Nageshwar_InturiG.pdf", "/Users/discovery/Desktop/rag_chatbot/docs/Nikhil_Nageshwar_Inturi_cr.pdf"])))
# print(doc_loader.get_document_stats(doc_loader.load_documents(["/Users/discovery/Desktop/rag_chatbot/docs/Nikhil_Nageshwar_InturiG.pdf", ])))