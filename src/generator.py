import time
import tiktoken
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGGenerator:
    def __init__(self, llm: BaseChatModel, model_name: str = "gpt-4o-mini"):
        self.llm = llm
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
        # Strict instructions minimize hallucination and improve Faithfulness
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Answer the user's question solely based on the provided context. If the context does not contain the necessary information, explicitly state 'I cannot answer this based on the context.' Do not use outside knowledge. Keep your answer concise and direct."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def count_tokens(self, text: str) -> int:
        """Helper to compute exact token counts for the selected model."""
        return len(self.tokenizer.encode(text))

    def generate(self, question: str, context_docs: List[Document]) -> Dict[str, Any]:
        """Generates an answer from context and computes token efficiency."""
        context_str = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Compute generation time strictly
        start_time = time.time()
        answer = self.chain.invoke({
            "context": context_str,
            "question": question
        })
        generation_time = time.time() - start_time
        
        # Proxy for conciseness (Metric #7)
        context_tokens = self.count_tokens(context_str)
        answer_tokens = self.count_tokens(answer)
        token_efficiency = answer_tokens / context_tokens if context_tokens > 0 else 0.0
        
        return {
            "answer": answer,
            "generation_time": generation_time,
            "context_tokens": context_tokens,
            "answer_tokens": answer_tokens,
            "token_efficiency": token_efficiency
        }
