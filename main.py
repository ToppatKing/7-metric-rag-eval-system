import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from src.config import RAGConfig
from src.indexer import Indexer, get_corpus_files
from src.evaluator import Evaluator
from src.generator import RAGGenerator
from src.retriever import RetrievalEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAGEvaluatorMain")

class StrategyRetrieverWrapper:
    """Adapts RetrievalEngine to LangChain's .invoke() interface for the generator."""
    def __init__(self, engine: RetrievalEngine, strategy_name: str):
        self.engine = engine
        self.strategy_name = strategy_name

    def invoke(self, query: str):
        return self.engine.retrieve(query, strategy=self.strategy_name)

def initialize_models(config: RAGConfig):
    """Initialize LLM and Embedding instances using RAGConfig."""
    if config.azure_api_key and config.azure_endpoint:
        logger.info("Initializing Azure OpenAI models...")
        llm = AzureChatOpenAI(
            azure_deployment=config.llm_model,
            api_version=config.azure_api_version,
            temperature=0.0,
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config.embedding_model,
            api_version=config.azure_api_version,
        )
        return llm, embeddings, f"azure:{config.llm_model}", f"azure:{config.embedding_model}"
    elif config.openai_api_key:
        logger.info("Initializing standard OpenAI models...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return llm, embeddings, "openai:gpt-4o-mini", "openai:text-embedding-3-small"
    else:
        logger.error("No valid API credentials found.")
        sys.exit(1)

def parse_args(config: RAGConfig):
    parser = argparse.ArgumentParser(description="7-Metric RAG Evaluation System")
    parser.add_argument("--corpus_dir", type=str, default="data", help="Directory containing source documents.")
    parser.add_argument("--persist_dir", type=str, default="chroma_db", help="Directory for vector DB storage.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for evaluation outputs.")
    parser.add_argument("--run_name", type=str, default="cuad_benchmark", help="Prefix label for this run.")
    parser.add_argument("--force_reindex", action="store_true", help="Force rebuild of the vector database.")
    
    args = parser.parse_args()
    return args

def main():
    config = RAGConfig()
    args = parse_args(config)
    
    logger.info("Starting RAG System Evaluation with Phase 4 Statistical Controls...")

    corpus_path = Path(args.corpus_dir)
    discovered_files = get_corpus_files(corpus_path)
    
    if not discovered_files:
        logger.error(f"No valid corpus documents found in '{corpus_path}'. Aborting.")
        sys.exit(1)

    llm, embeddings, model_name, embedding_name = initialize_models(config)

    indexer = Indexer(
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
        embedding_model_name=embedding_name,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
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

    # Initialize the custom RetrievalEngine
    retrieval_engine = RetrievalEngine(vectorstore=vectorstore, llm=llm, top_k=config.top_k)

    strategies = {
        "Dense_Similarity": StrategyRetrieverWrapper(retrieval_engine, "dense"),
        "MMR_Search": StrategyRetrieverWrapper(retrieval_engine, "mmr"),
        "HyDE_Search": StrategyRetrieverWrapper(retrieval_engine, "hyde"),
    }

    generator = RAGGenerator(llm=llm, model_name=model_name)
    evaluator = Evaluator(llm=llm, embeddings=embeddings, output_dir=args.results_dir)

    strategy_trial_outputs: Dict[str, List[List[Dict[str, Any]]]] = {s: [] for s in strategies}

    for trial_idx in range(config.num_trials):
        logger.info(f"=== Starting Trial Execution {trial_idx + 1}/{config.num_trials} ===")
        
        strategy_items = list(strategies.items())
        random.shuffle(strategy_items)

        for strategy_name, retriever in strategy_items:
            single_trial_run = []
            for question in test_questions:
                run_output = generator.run_pipeline(retriever, question)
                single_trial_run.append(run_output)
            
            strategy_trial_outputs[strategy_name].append(single_trial_run)

    final_strategy_results = []
    for strategy_name in strategies:
        trial_data = strategy_trial_outputs[strategy_name]
        eval_summary = evaluator.evaluate_multi_trial_strategy(
            strategy_name=strategy_name,
            trial_outputs=trial_data,
            ground_truths=ground_truths
        )
        final_strategy_results.append(eval_summary)

    config_manifest = {
        "llm_model": model_name,
        "embedding_model": embedding_name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "num_trials": config.num_trials,
        "total_documents": len(discovered_files),
    }

    output_path = evaluator.write_artifacts(
        run_name=args.run_name,
        experiment_config=config_manifest,
        strategy_results=final_strategy_results
    )

    logger.info(f"Benchmark successfully completed. Results stored in: {output_path}")

if __name__ == "__main__":
    main()
