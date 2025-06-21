"""
Retrieval logic

# maximum marginal relevance
# https://www.kaggle.com/code/marcinrutecki/rag-mmr-search-in-langchain
"""

# import
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from utils.vectorstore import VectorStore
# logging
logger = logging.getLogger(__name__)

class Retriever:
    """Handles document retrieval with various strategies"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve_documents(self, query: str, method: str = "similarity", k: int = 5, filter_criteria: Optional[Dict] = None, score_threshold: Optional[float] = None) -> List[Document]:
        """
        Retrieve documents using specified method
        
        Args:
            query: Search query
            method: Retrieval method ('similarity', 'mmr')
            k: Number of documents to retrieve
            filter_criteria: Metadata filters
            score_threshold: Minimum similarity score
        """
        try:
            if method == "similarity":
                if score_threshold:
                    results = self.vector_store.similarity_search_with_score(query, k=k*2)
                    filtered_docs = [doc for doc, score in results if score <= score_threshold][:k]
                else:
                    filtered_docs = self.vector_store.similarity_search(query, k=k, filter_dict=filter_criteria)
            elif method == "mmr":  
                retriever = self.vector_store.get_retriever(search_type="mmr", search_kwargs={"k": k, "filter": filter_criteria})
                filtered_docs = retriever.get_relevant_documents(query)
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
            logger.info(f"Retrieved {len(filtered_docs)} documents using {method}")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []
    
    def retrieve_with_context(self, query: str, context_window: int = 1, **kwargs) -> List[Document]:
        """
        Retrieve documents with surrounding context chunks
        
        Args:
        query: Search query
        context_window: Number of chunks to include before and after each retrieved chunk
        **kwargs: Additional parameters passed to the underlying retrieval method, e.g.:
            method: Retrieval method ('similarity', 'mmr')
            k: Number of documents to retrieve
            filter_criteria: Metadata filters
            score_threshold: Minimum similarity score
        """
        docs = self.retrieve_documents(query, **kwargs)
        return docs
    
    def hybrid_retrieve(self, query: str,keyword_weight: float = 0.3, semantic_weight: float = 0.7, k: int = 5) -> List[Document]:
        """
        Perform a hybrid retrieval combining keyword-based and semantic search.

        Args:
            query: Search query string.
            keyword_weight: Weight to assign to keyword-based matching (0.0–1.0).
            semantic_weight: Weight to assign to semantic similarity matching (0.0–1.0).
            k: Total number of documents to return after combining both scores.
        """
        return self.retrieve_documents(query, k=k)