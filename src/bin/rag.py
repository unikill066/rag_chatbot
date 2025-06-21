"""
Main RAG chatbot orchestrating all components
"""

import os
import logging
from typing import Dict, Any, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from utils.document_loader import DocumentLoader
from utils.text_processor import TextProcessor
from utils.vectorstore import VectorStore
from utils.retriever import Retriever
from constants import SUPPORTED_EXTENSIONS

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    query: str
    query_type: str
    retrieved_docs: List[Any]
    context: str
    response: str
    suggested_questions: List[str]
    error: str

class RAGChatbot:
    """Main RAG chatbot class"""
    
    def __init__(self,vector_store: str = "vector_store", model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the RAG chatbot
        
        Args:
            vector_store: Path to the vector store
            model_name: Name of the LLM model
        """
        self.document_loader = DocumentLoader(SUPPORTED_EXTENSIONS)
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore(vector_store)
        self.retriever = Retriever(self.vector_store)
        self.llm = ChatOpenAI(model=model_name, temperature=0.5)
    
        self.query_patterns = {
            "summarize": ["summarize", "summary", "brief", "overview"],
            "research": ["research", "investigate", "analyze", "study"],
            "explain": ["explain", "what is", "how does", "define"],
            "compare": ["compare", "difference", "versus", "vs"],
            "list": ["list", "enumerate", "what are", "types"],
            "general": []
        }
        
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())  # checkpointer to cache in-memory
    
    def add_documents(self, file_paths: List[str]):
        """
        Add documents to the system
        
        Args:
            file_paths: List of file paths to add
        """
        try:
            documents = self.document_loader.load_documents(file_paths)
            chunks = self.text_processor.chunk_documents(documents)
            self.vector_store.add_documents(chunks)
            logger.info(f"Successfully added {len(file_paths)} files with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def add_directory(self, directory_path: str, file_types: List[str] = None):
        """
        Add all documents from a directory
        
        Args:
            directory_path: Path to the directory
            file_types: List of file types to include
        """
        try:
            documents = self.document_loader.load_directory(directory_path, file_types)
            chunks = self.text_processor.chunk_documents(documents)
            self.vector_store.add_documents(chunks)
            logger.info(f"Added directory {directory_path} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding directory: {e}")
            raise
    
    def _classify_query(self, query: str) -> str:
        """
        Classify query type
        
        Args:
            query: Query to classify
        """
        query_lower = query.lower()
        for query_type, patterns in self.query_patterns.items():
            if query_type == "general":
                continue
            for pattern in patterns:
                if pattern in query_lower:
                    return query_type
        return "general"
    
    def _get_prompt_template(self, query_type: str) -> ChatPromptTemplate:
        """
        Get prompt template based on query type
        
        Args:
            query_type: Query type
        """
        templates = {
            "summarize": """Provide a clear, concise summary based on the context.
Context: {context}
Question: {question}
Summary:""",
            
            "research": """Provide a comprehensive research-based analysis.
Context: {context}
Question: {question}
Analysis:""",
            
            "explain": """Explain the topic clearly and thoroughly.
Context: {context}
Question: {question}
Explanation:""",
            
            "compare": """Provide a detailed comparison.
Context: {context}
Question: {question}
Comparison:""",
            
            "list": """Provide a well-organized list.
Context: {context}
Question: {question}
List:""",
            
            "general": """Answer the question based on the provided context.
Context: {context}
Question: {question}
Answer:"""
        }
        return ChatPromptTemplate.from_template(templates.get(query_type, templates["general"]))
    
    def _create_workflow(self) -> StateGraph:
        """
        Create LangGraph workflow
        """
        def classify_query_node(state: RAGState) -> RAGState:
            state["query_type"] = self._classify_query(state["query"])
            return state
        
        def retrieve_documents_node(state: RAGState) -> RAGState:
            try:
                docs = self.retriever.retrieve_documents(state["query"], k=5)
                state["retrieved_docs"] = docs
                state["context"] = "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                state["error"] = str(e)
            return state
        
        def generate_response_node(state: RAGState) -> RAGState:
            try:
                if state.get("error"):
                    state["response"] = f"Error: {state['error']}"
                    return state
                prompt = self._get_prompt_template(state["query_type"])
                chain = prompt | self.llm | StrOutputParser()
                state["response"] = chain.invoke({"context": state["context"], "question": state["query"]})
            except Exception as e:
                state["error"] = str(e)
                state["response"] = f"Error generating response: {e}"
            return state
        
        def generate_suggestions_node(state: RAGState) -> RAGState:
            try:
                suggestion_prompt = ChatPromptTemplate.from_template(
                    """Based on this query and context, suggest 3 follow-up questions:
                    Query: {query}
                    Context: {context}
                    
                    Provide 3 questions, one per line:""")
                
                chain = suggestion_prompt | self.llm | StrOutputParser()
                suggestions = chain.invoke({"query": state["query"], "context": state["context"][:500]})
                
                questions = [q.strip() for q in suggestions.split('\n') if q.strip()]
                state["suggested_questions"] = questions[:3]
                
            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")
                state["suggested_questions"] = [] 
            return state
        
        workflow = StateGraph(RAGState)
        workflow.add_node("classify", classify_query_node)
        workflow.add_node("retrieve", retrieve_documents_node)
        workflow.add_node("generate", generate_response_node)
        workflow.add_node("suggest", generate_suggestions_node)
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "suggest")
        workflow.add_edge("suggest", END)
        return workflow
    
    def query(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            thread_id: Thread ID
        """
        try:
            initial_state = {"query": question, "query_type": "", "retrieved_docs": [], "context": "",
                "response": "", "suggested_questions": [], "error": ""}
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config=config)
            return {"query": result["query"], "query_type": result["query_type"],
                  "response": result["response"], "suggested_questions": result["suggested_questions"],
                  "retrieved_docs_count": len(result["retrieved_docs"])}
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"query": question, "response": f"Error processing query: {e}",
                "query_type": "error", "suggested_questions": [], "retrieved_docs_count": 0}
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information
        """
        return {"vector_store_info": self.vector_store.get_store_info(),
               "supported_file_types": list(DocumentLoader.SUPPORTED_EXTENSIONS.keys())}