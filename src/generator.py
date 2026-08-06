import time
import logging
from typing import List, Dict, Any, Tuple
import tiktoken

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

# Standard Azure / OpenAI Pricing per 1,000,000 Tokens (USD)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "default": {"input": 0.15, "output": 0.60},
}


class RAGGenerator:
    """
    RAG Generator that measures retrieval and generation timings,
    counts exact tokens, estimates API cost, and calculates compression ratios.
    """

    def __init__(self, llm: BaseLanguageModel, model_name: str = "gpt-4o-mini"):
        self.llm = llm
        self.model_name = model_name
        
        # Initialize token counter (defaults to cl100k_base for GPT-3.5/4/4o)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        self.prompt_template = ChatPromptTemplate.from_template(
            "You are an expert assistant evaluating legal documents.\n"
            "Answer the user's question accurately using ONLY the following retrieved contexts.\n"
            "If the answer is not contained in the context, state 'I cannot answer based on the context.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken, falling back to word estimation if uninitialized."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text.split()) * 1.3)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated API cost in USD."""
        pricing = MODEL_PRICING.get(self.model_name.replace("azure:", "").replace("openai:", ""), MODEL_PRICING["default"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def run_pipeline(
        self,
        retriever: Any,
        question: str
    ) -> Dict[str, Any]:
        """
        Executes a single RAG pipeline run, recording fine-grained performance metrics.
        """
        # 1. Measure Retrieval Timing
        start_retrieval = time.perf_counter()
        docs = retriever.invoke(question)
        retrieval_latency = time.perf_counter() - start_retrieval

        chunk_texts = [d.page_content for d in docs]
        chunk_ids = [str(d.metadata.get("source", "unknown")) + f"#page={d.metadata.get('page', 0)}" for d in docs]
        context_block = "\n\n".join(chunk_texts)

        # 2. Measure Generation Timing
        start_generation = time.perf_counter()
        answer = self.chain.invoke({"context": context_block, "question": question})
        generation_latency = time.perf_counter() - start_generation

        total_latency = retrieval_latency + generation_latency

        # 3. Token Accounting
        context_tokens = self._count_tokens(context_block)
        answer_tokens = self._count_tokens(answer)
        prompt_tokens = self._count_tokens(
            f"You are an expert assistant evaluating legal documents...\nContext:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
        )

        # 4. Correct Metric Naming: Compression Ratio (Answer Tokens / Context Tokens)
        compression_ratio = round(answer_tokens / max(context_tokens, 1), 4)
        estimated_cost = self._estimate_cost(prompt_tokens, answer_tokens)

        return {
            "question": question,
            "answer": answer,
            "contexts": chunk_texts,
            "retrieved_doc_ids": chunk_ids,
            "metrics": {
                "retrieval_latency_sec": round(retrieval_latency, 4),
                "generation_latency_sec": round(generation_latency, 4),
                "total_latency_sec": round(total_latency, 4),
                "context_tokens": context_tokens,
                "answer_tokens": answer_tokens,
                "prompt_tokens": prompt_tokens,
                "context_to_answer_compression_ratio": compression_ratio,
                "estimated_cost_usd": estimated_cost,
            }
        }
