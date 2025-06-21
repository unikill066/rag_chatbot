"""
Vector store management with FAISS
# https://python.langchain.com/docs/integrations/vectorstores/faiss/
"""

# imports
import os, logging, pickle, faiss, shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# logging
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages FAISS vector store operations"""

    def __init__(self, store_path: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize vector store manager
        
        Args:
            store_path: Path to store vector database
            embedding_model: Name of embedding model if embeddings not provided
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self._load_or_create_store()

    def _load_or_create_store(self):
        """Load existing store or create new one"""
        faiss_path = self.store_path / "faiss_index"
        
        if faiss_path.exists():
            try:
                self.vector_store = FAISS.load_local(str(self.store_path), self.embeddings, index_name="faiss_index")
                logger.info("Loaded existing vector store")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing store: {e}")

        logger.info("Creating new vector store")
        dummy_doc = Document(page_content="Initial document for vector store setup", metadata={"source": "system", "type": "research"})
        self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
        try:
            self.vector_store.save_local(str(self.store_path), index_name="faiss_index")
        except Exception as e:
            logger.error(f"Failed to save vectore store: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add a list of documents to the vector store and return their assigned IDs.

        Args:
            documents: List of Document objects to be added to the vector store."""
        if not isinstance(documents, list):
            logging.error(f"Documents are not in expected list format: {documents}")
            raise
        if not documents:
            logger.warning("No documents to add")
            raise
        try:
            if self._is_dummy_store():
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info("Replaced dummy store with real documents")
            else:
                ids = self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
            
            try:
                self.vector_store.save_local(str(self.store_path), index_name="faiss_index")
            except Exception as e:
                logger.error(f"Failed to save vectore store: {e}")
                raise
            return self.vector_store.index_to_docstore_id.values()
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def _is_dummy_store(self) -> bool:
        """Check if store contains only research document"""
        if not hasattr(self.vector_store, 'docstore'):
            return False
        
        docs = list(self.vector_store.docstore._dict.values())
        return (len(docs) == 1 and 
                docs[0].metadata.get("type") == "research")

    def similarity_search(self, query: str, k: int = 4, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if filter_dict:
                docs = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} documents for query")
            return docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search for documents most similar to the given query using vector embeddings.

        Args:
            query: Text query to compare against the vector index.
            k: Number of top-matching documents to return.
            filter_dict: Optional metadata-based filters to apply before searching
                        (e.g., {"source": "news"}).
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            return []

    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        try:
            num_docs = len(self.vector_store.docstore._dict) if hasattr(self.vector_store, 'docstore') else 0
            index_size = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            return {"status": "Success", "num_documents": num_docs, "index_size": index_size,"store_path": str(self.store_path)}
        
        except Exception as e:
            return {"status": "error", "error": str(e)}

# # testing and generate indexes
# # python vectorstore.py
# if __name__ == '__main__':
#     document_directory = Path("/Users/discovery/Desktop/rag_chatbot/docs")
#     if not document_directory.exists():
#         raise ValueError("Document directory does not exist")
#     from document_loader import DocumentLoader
#     from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
#     doc_loader = DocumentLoader({'.pdf': PyPDFLoader, '.txt': TextLoader, '.csv': CSVLoader,})
#     documents = doc_loader.load_directory(document_directory)
#     print(f"Loaded {len(documents)} documents from {document_directory}")
#     from text_processor import TextProcessor
#     text_proc = TextProcessor(chunk_size=400, chunk_overlap=100, splitter_type="recursive")
#     chunks = text_proc.chunk_documents(documents)
#     print(f"Chunked {len(documents)} documents into {len(chunks)} chunks")

#     from dotenv import load_dotenv
#     load_dotenv()
#     store_path = Path(__file__).parent.parent.parent / "faiss_vector_store"  # or import from constants.py file
#     # generate fresh embeddings
#     if store_path.exists():
#         import shutil; shutil.rmtree(store_path)
#     vec_store = VectorStore(store_path=store_path, embedding_model="text-embedding-3-small")
#     vec_store.add_documents(chunks)
#     print(f"Added {len(chunks)} chunks to vector store: {store_path}")
#     print(vec_store.get_store_info())