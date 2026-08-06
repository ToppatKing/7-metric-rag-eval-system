import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from src.indexer import Indexer, get_corpus_files
from src.evaluator import Evaluator
from src.generator import RAGGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAGEvaluatorMain")

load_dotenv()


def initialize_models():
    """Initialize LLM and Embedding instances."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="7-Metric RAG Evaluation System")
    parser.add_argument("--corpus_dir", type=str, default="data", help="Directory containing source documents.")
    parser.add_argument("--persist_dir", type=str, default="chroma_db", help="Directory for vector DB storage.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for evaluation outputs.")
    parser.add_argument("--run_name", type=str, default="cuad_benchmark", help="Prefix label for this run.")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of repeated trials per strategy.")
    parser.add_argument("--force_reindex", action="store_true", help="Force rebuild of the vector database.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Document chunk size.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Document chunk overlap.")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve per query.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting RAG System Evaluation with Phase 4 Statistical Controls...")

    corpus_path = Path(args.corpus_dir)
    discovered_files = get_corpus_files(corpus_path)
    
    if not discovered_files:
        logger.error(f"No valid corpus documents found in '{corpus_path}'. Aborting.")
        sys.exit(1)

    llm, embeddings, model_name, embedding_name = initialize_models()

    # Indexing Phase
    indexer = Indexer(
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
        embedding_model_name=embedding_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    vectorstore = indexer.build_or_load_index(
        corpus_directory=corpus_path,
        force_reindex=args.force_reindex
    )

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

    strategies = {
        "Dense_Similarity": vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": args.top_k}
        ),
        "MMR_Search": vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": args.top_k, "fetch_k": args.top_k * 3, "lambda_mult": 0.5}
        ),
    }

    generator = RAGGenerator(llm=llm, model_name=model_name)
    evaluator = Evaluator(llm=llm, embeddings=embeddings, output_dir=args.results_dir)

    strategy_trial_outputs: Dict[str, List[List[Dict[str, Any]]]] = {s: [] for s in strategies}

    # P2 Fix: Execute N trials with randomized execution order to eliminate API warm-up bias
    for trial_idx in range(args.num_trials):
        logger.info(f"=== Starting Trial Execution {trial_idx + 1}/{args.num_trials} ===")
        
        strategy_items = list(strategies.items())
        random.shuffle(strategy_items)

        for strategy_name, retriever in strategy_items:
            single_trial_run = []
            for question in test_questions:
                run_output = generator.run_pipeline(retriever, question)
                single_trial_run.append(run_output)
            
            strategy_trial_outputs[strategy_name].append(single_trial_run)

    # Evaluate strategies using statistical controls
    final_strategy_results = []
    for strategy_name in strategies:
        trial_data = strategy_trial_outputs[strategy_name]
        eval_summary = evaluator.evaluate_multi_trial_strategy(
            strategy_name=strategy_name,
            trial_outputs=trial_data,
            ground_truths=ground_truths
        )
        final_strategy_results.append(eval_summary)

    # Write auditable artifacts
    config_manifest = {
        "llm_model": model_name,
        "embedding_model": embedding_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "num_trials": args.num_trials,
        "total_documents": len(discovered_files),
    }

    output_path = evaluator.write_artifacts(
        run_name=args.run_name,
        experiment_config=config_manifest,
        strategy_results=final_strategy_results
    )

    logger.info(f"Phase 4 Benchmark successfully completed. Results stored in: {output_path}")


if __name__ == "__main__":
    main()
