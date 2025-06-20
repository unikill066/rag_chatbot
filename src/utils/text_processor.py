"""
Text processing and chunking utilities


Reference(s): https://python.langchain.com/docs/concepts/text_splitters/
"""


# imports
import logging
from typing import List, Dict, Any
from langchain.text_splitter import (RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter)
from langchain_core.documents import Document
from pydantic.v1 import EnumMemberError

# logging
logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text chunking and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, splitter_type: str = "recursive"):
        """
        Initialize text processor

        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            splitter_type: Type of splitter ('recursive', 'character', 'token')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = self._create_splitter(splitter_type)

    def _create_splitter(self, splitter_type: str):
        """Create appropriate text splitter"""
        splitters = {
            "recursive": RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len, separators=["\n\n", "\n"," ", ""]),
            "character": CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n"),
            "token"    : TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)            
        }

        if splitter_type not in splitters:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
        return splitters[splitter_type]

    def preprocess_text(self, text: str) -> str:
        """Preprocess text (clean, normalize)"""
        return ' '.join(text.strip().split())

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunks documents into smaller chunks"""
        if not isinstance(documents, list):
            logger.warning("No documents to chunk")
            raise TypeError("Expected a list of Document instances")

        if len(documents) == 0:
            logger.warning("No documents to chunk")
            raise ValueError("The documents list is empty")
        
        try:
            chunks = self.splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({'chunk_id': 1, 'chunk_size': len(chunk.page_content)})
            
            logger.info(f"Created {len(chunks)} chinks from {len(documents)} documents")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise