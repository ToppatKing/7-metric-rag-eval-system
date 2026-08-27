import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, top_k: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

    def retrieve_dense(self, query: str) -> List[Document]:
        """Standard semantic search using Dense Vectors."""
        logger.info("Executing Dense Vector Retrieval...")
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.top_k}
        )
        return retriever.invoke(query)

    def retrieve_mmr(self, query: str) -> List[Document]:
        """Maximal Marginal Relevance (MMR) for diverse context fetching."""
        logger.info("Executing MMR Retrieval...")
        retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            # Fetch 20 chunks initially, then pick the 5 most diverse
            search_kwargs={"k": self.top_k, "fetch_k": self.top_k * 4, "lambda_mult": 0.5}
        )
        return retriever.invoke(query)

def retrieve_hyde(self, query: str) -> List[Document]:
        """Hypothetical Document Embeddings (HyDE) strategy with Anchoring and Constraints."""
        logger.info("Executing HyDE Retrieval...")
        
        # Step 1: Instruct the LLM to write a constrained hypothetical response
        # We add CRITICAL INSTRUCTIONS to prevent the "Hallucination Anchor"
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly skilled domain expert. 
Write a hypothetical excerpt from a reference document (like a legal contract or a historical museum record) that would perfectly answer the user's question. 

CRITICAL INSTRUCTIONS:
1. Use standard domain boilerplate, jargon, and terminology.
2. DO NOT invent specific states, locations, names, dates, notice periods, or quantitative numbers. 
3. Keep it under 3 sentences."""),
            ("human", "{question}")
        ])
        
        chain = hyde_prompt | self.llm | StrOutputParser()
        hypothetical_doc = chain.invoke({"question": query})
        logger.debug(f"Hypothetical Document Generated:\n{hypothetical_doc}")
        
        # Step 2: ANCHOR THE SEARCH
        # Combine the exact original query with the hypothetical text so the mathematical 
        # vector retains the user's original intent.
        anchored_search_query = f"Original Question: {query}\n\nRelevant Excerpt: {hypothetical_doc}"
        
        # Step 3: Use the anchored query to search the vector store
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.top_k}
        )
        
        # We embed the anchored query, not just the hallucinated text
        return retriever.invoke(anchored_search_query)
        
    def retrieve(self, query: str, strategy: str = "dense") -> List[Document]:
        """Routing method for the retrieval strategy."""
        strategies = {
            "dense": self.retrieve_dense,
            "mmr": self.retrieve_mmr,
            "hyde": self.retrieve_hyde
        }
        
        if strategy not in strategies:
            raise ValueError(f"Strategy must be one of: {list(strategies.keys())}")
            
        return strategies[strategy](query)
