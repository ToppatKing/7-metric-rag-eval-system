import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

# Local Module Imports
from src.indexer import Indexer, get_corpus_files
from src.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAGEvaluatorMain")

# Load environment variables from .env file
load_dotenv()


def initialize_models():
    """
    Initialize LLM and Embedding instances.
    Prefers Azure OpenAI if environment variables are set, 
    otherwise falls back to standard OpenAI.
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    if azure_key and azure_endpoint:
        logger.info("Initializing Azure OpenAI models...")
        llm = AzureChatOpenAI(
            azure_deployment=chat_deployment,
            api_version=azure_api_version,
            temperature=0.0,
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            api_version=azure_api_version,
        )
        model_name = f"azure:{chat_deployment}"
        embedding_name = f"azure:{embedding_deployment}"
    elif os.getenv("OPENAI_API_KEY"):
        logger.info("Initializing standard OpenAI models...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        model_name = "openai:gpt-4o-mini"
        embedding_name = "openai:text-embedding-3-small"
    else:
        logger.error("No valid API credentials found in environment variables (.env).")
        sys.exit(1)

    return llm, embeddings, model_name, embedding_name


def generate_rag_responses(
    llm,
    retriever,
    questions: List[str]
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    Executes RAG pipeline for a set of questions using a given retriever.
    Returns generated answers, retrieved text contexts, and source document metadata IDs.
    """
    prompt_template = ChatPromptTemplate.from_template(
        "You are an expert assistant evaluating legal documents.\n"
        "Answer the user's question accurately using ONLY the following retrieved contexts.\n"
        "If the answer is not contained in the context, state 'I cannot answer based on the context.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt_template | llm | StrOutputParser()

    answers = []
    contexts = []
    retrieved_doc_ids = []

    for q in questions:
        # Retrieve relevant chunks using modern LangChain invoke API
        docs = retriever.invoke(q)
        
        chunk_texts = [d.page_content for d in docs]
        chunk_ids = [str(d.metadata.get("source", "unknown")) + f"#page={d.metadata.get('page', 0)}" for d in docs]
        
        context_block = "\n\n".join(chunk_texts)
        
        # Generate answer
        answer = chain.invoke({"context": context_block, "question": q})

        answers.append(answer)
        contexts.append(chunk_texts)
        retrieved_doc_ids.append(chunk_ids)

    return answers, contexts, retrieved_doc_ids


def parse_args():
    """Parse Command-Line Arguments."""
    parser = argparse.ArgumentParser(description="7-Metric RAG Evaluation System")
    parser.add_argument("--corpus_dir", type=str, default="data", help="Directory containing source documents.")
    parser.add_argument("--persist_dir", type=str, default="chroma_db", help="Directory for vector DB storage.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for evaluation outputs.")
    parser.add_argument("--run_name", type=str, default="cuad_benchmark", help="Prefix label for this run.")
    parser.add_argument("--force_reindex", action="store_true", help="Force rebuild of the vector database.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Document chunk size in characters.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Document chunk overlap in characters.")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve per query.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting RAG System Evaluation...")

    # 1. P3 Fix: Unified file discovery validation
    corpus_path = Path(args.corpus_dir)
    discovered_files = get_corpus_files(corpus_path)
    
    if not discovered_files:
        logger.error(f"No valid corpus documents found in directory '{corpus_path}'. Aborting.")
        sys.exit(1)
        
    logger.info(f"Discovered {len(discovered_files)} supported corpus document(s) in '{corpus_path}'.")

    # 2. Initialize Models (P1 Security & P2 Explicit Dependencies)
    llm, embeddings, model_name, embedding_name = initialize_models()

    # 3. Build or Load Manifest-Backed Vector Database (P1 Index Integrity)
    indexer = Indexer(
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
        embedding_model_name=embedding_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    try:
        vectorstore = indexer.build_or_load_index(
            corpus_directory=corpus_path,
            force_reindex=args.force_reindex
        )
    except Exception as e:
        logger.critical(f"Indexing phase failed: {e}")
        sys.exit(1)

    # 4. Define Test Set (Example CUAD / Legal Questions & Ground Truths)
    # In production, this can be loaded from a json/csv file
    test_questions = [
        "What is the governing law specified in the agreement?",
        "What are the termination rights and notice periods for breach?",
        "Does the agreement contain an exclusivity or non-compete clause?"
    ]
    ground_truths = [
        ["The agreement is governed by the laws of the State of Delaware."],
        ["Either party may terminate upon 30 days written notice of a material breach."],
        ["Yes, Section 4 contains an exclusive distribution rights clause."]
    ]

    # 5. Define Retrieval Strategies to Compare
    strategies = {
        "Dense_Similarity": vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.top_k}
        ),
        "MMR_Search": vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": args.top_k, "fetch_k": args.top_k * 3, "lambda_mult": 0.5}
        ),
    }

    # 6. Initialize Evaluator (P2 Explicit Dependency Injection)
    evaluator = Evaluator(
        llm=llm,
        embeddings=embeddings,
        output_dir=args.results_dir
    )

    strategy_results = []

    # 7. Run Generation and Evaluation per Strategy
    for strategy_name, retriever in strategies.items():
        logger.info(f"--- Running Strategy Pipeline: {strategy_name} ---")
        
        # Generation phase
        answers, contexts, doc_ids = generate_rag_responses(
            llm=llm,
            retriever=retriever,
            questions=test_questions
        )

        # Evaluation phase (P1 Exception Handling & Metric Validation)
        eval_result = evaluator.evaluate_strategy(
            strategy_name=strategy_name,
            questions=test_questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            retrieved_doc_ids=doc_ids,
        )
        
        strategy_results.append(eval_result)

    # 8. Store Auditable, Timestamped Experiment Manifest & Results (P1 Auditability)
    config_manifest = {
        "llm_model": model_name,
        "embedding_model": embedding_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "corpus_dir": str(corpus_path),
        "total_documents": len(discovered_files),
    }

    output_path = evaluator.write_artifacts(
        run_name=args.run_name,
        experiment_config=config_manifest,
        strategy_results=strategy_results
    )

    logger.info(f"Evaluation process completed successfully. Final results stored in: {output_path}")


if __name__ == "__main__":
    main()
