import os
import sys
import shutil
import argparse
import logging
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Silence ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
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
            chunk_size=16,       # Sends 16 chunks per API request
            max_retries=20,      # Retries automatically if temporary rate limits occur
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

def load_museum_csv_data(data_dir: str) -> List[Document]:
    """Reads all CSV files in data_dir and converts spreadsheet rows into formatted Documents."""
    documents = []
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory '{data_dir}' not found.")
        return documents

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                
                for idx, row in df.iterrows():
                    def get_col(name):
                        return str(row[name]) if name in df.columns and pd.notna(row[name]) else "N/A"
                    
                    content = (
                        f"Museum: {get_col('museo')}\n"
                        f"Artwork: {get_col('titolo')}\n"
                        f"Artist/Producer: {get_col('autore')}\n"
                        f"Estimated Date: {get_col('datazione')}\n"
                        f"Type: {get_col('tipologia')}\n"
                        f"Subject: {get_col('soggetto')}\n"
                        f"Materials/Technique: {get_col('materiale_tecnica')}\n"
                        f"Dimensions: {get_col('misure')}\n"
                        f"Conservation Location: {get_col('luogo_conservazione')}\n"
                        f"Localization: {get_col('localizzazione')}\n"
                        f"Address: {get_col('indirizzo')}\n"
                        f"Historian/Critic Notes: {get_col('note_storico_critiche')}"
                    )
                    
                    metadata = {
                        "source_file": file,
                        "museum": get_col('museo'),
                        "artwork_name": get_col('titolo')
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
                
    logger.info(f"Loaded {len(documents)} artwork records from CSV files.")
    return documents

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Dataset RAG Evaluation System")
    parser.add_argument("--dataset", type=str, choices=["cuad", "museums"], default="cuad", help="Select dataset to evaluate.")
    parser.add_argument("--mode", type=str, choices=["benchmark", "interactive"], default="benchmark", help="Execution mode.")
    parser.add_argument("--corpus_dir", type=str, default=None, help="Override default data directory.")
    parser.add_argument("--persist_dir", type=str, default=None, help="Override default vector DB storage.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for evaluation outputs.")
    parser.add_argument("--run_name", type=str, default="cuad_benchmark", help="Prefix label for artifacts.")
    parser.add_argument("--force_reindex", action="store_true", help="Force rebuild of vector database.")
    
    args = parser.parse_args()
    return args

def main():
    config = RAGConfig()
    args = parse_args()
    
    # Set default paths based on selected dataset
    if args.dataset == "cuad":
        corpus_dir = args.corpus_dir or "data"
        persist_dir = args.persist_dir or "chroma_db_cuad"
    else:
        corpus_dir = args.corpus_dir or "data_museums"
        persist_dir = args.persist_dir or "chroma_db_museums"

    logger.info(f"Starting RAG System [{args.dataset.upper()} Dataset | {args.mode.upper()} Mode]...")

    # Initialize Azure Models
    llm, embeddings, model_name, embedding_name = initialize_models(config)

    # 1. Build or Load Index
    if args.dataset == "cuad":
        corpus_path = Path(corpus_dir)
        discovered_files = get_corpus_files(corpus_path)
        if not discovered_files:
            logger.error(f"No valid corpus documents found in '{corpus_path}'. Aborting.")
            sys.exit(1)

        indexer = Indexer(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            embedding_model_name=embedding_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        vectorstore = indexer.build_or_load_index(
            corpus_directory=corpus_path,
            force_reindex=args.force_reindex
        )
        total_docs = len(discovered_files)
    else:
        raw_docs = load_museum_csv_data(corpus_dir)
        if not raw_docs:
            logger.error(f"No valid museum records found in '{corpus_dir}'. Aborting.")
            sys.exit(1)

        persist_path = Path(persist_dir)
        if args.force_reindex and persist_path.exists():
            logger.info(f"Force reindex requested. Purging database at: {persist_path}")
            shutil.rmtree(persist_path)

        if not persist_path.exists() or not os.listdir(persist_path):
            logger.info(f"Building new vector database from {len(raw_docs)} artwork records...")
            vectorstore = Chroma.from_documents(
                documents=raw_docs,
                embedding=embeddings,
                persist_directory=str(persist_path)
            )
        else:
            logger.info(f"Loading existing vector database from: {persist_path}")
            vectorstore = Chroma(
                persist_directory=str(persist_path),
                embedding_function=embeddings
            )
        total_docs = len(raw_docs)

    # 2. Setup Retrieval Engine and Strategies
    retrieval_engine = RetrievalEngine(vectorstore=vectorstore, llm=llm, top_k=config.top_k)
    strategies = {
        "Dense_Similarity": StrategyRetrieverWrapper(retrieval_engine, "dense"),
        "MMR_Search": StrategyRetrieverWrapper(retrieval_engine, "mmr"),
        "HyDE_Search": StrategyRetrieverWrapper(retrieval_engine, "hyde"),
    }
    generator = RAGGenerator(llm=llm, model_name=model_name)

    # 3. Execution Logic based on Mode
    if args.mode == "benchmark":
        evaluator = Evaluator(llm=llm, embeddings=embeddings, output_dir=args.results_dir)

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
            "total_documents": total_docs,
        }

        output_path = evaluator.write_artifacts(
            run_name=args.run_name,
            experiment_config=config_manifest,
            strategy_results=final_strategy_results
        )
        logger.info(f"Benchmark completed successfully. Artifacts stored in: {output_path}")

    elif args.mode == "interactive":
        print("\n" + "="*70)
        print(f"      {args.dataset.upper()} RAG INTERACTIVE MULTI-STRATEGY BENCHMARK")
        print("="*70)
        
        while True:
            query = input("\nEnter your question (or type 'exit' to quit):\n> ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Exiting interactive mode.")
                break
                
            if not query:
                continue
                
            ground_truth = input("\nEnter Ground Truth (optional, press Enter to skip):\n> ").strip()
            if not ground_truth:
                ground_truth = None
                
            print(f"\n[Thinking...] Running retrieval and generating responses across 3 strategies...")

            for strategy_name, retriever in strategies.items():
                print("\n" + "-"*60)
                print(f" STRATEGY: {strategy_name}")
                print("-" * 60)
                
                try:
                    run_output = generator.run_pipeline(retriever, query) 
                    
                    print("\n--- GENERATED RESPONSE ---")
                    print(run_output.get("answer", "No answer generated."))
                    
                    print("\n--- RETRIEVED CONTEXTS ---")
                    contexts = run_output.get("contexts", [])
                    for i, doc in enumerate(contexts, 1):
                        if isinstance(doc, str):
                            first_line = doc.split('\n')[0] if '\n' in doc else doc[:100]
                            second_line = doc.split('\n')[1] if '\n' in doc and len(doc.split('\n')) > 1 else ""
                            print(f"[{i}] {first_line} | {second_line}")
                        elif hasattr(doc, 'metadata'):
                            artwork = doc.metadata.get('artwork_name', 'Unknown Document')
                            museum = doc.metadata.get('museum', '')
                            print(f"[{i}] {artwork} ({museum})")
                        else:
                            print(f"[{i}] {str(doc)[:100]}...")
                    
                    metrics = run_output.get("metrics", {})
                    print("\n--- EXECUTION METRICS ---")
                    if "total_latency_sec" in metrics:
                        print(f"  Total Latency: {metrics['total_latency_sec']:.2f} seconds")
                    if "estimated_cost_usd" in metrics:
                        print(f"  Estimated Cost: ${metrics['estimated_cost_usd']:.6f}")
                    if "prompt_tokens" in metrics:
                        print(f"  Tokens Used: {metrics.get('prompt_tokens', 0)} prompt / {metrics.get('answer_tokens', 0)} answer")

                except Exception as e:
                    logger.error(f"Error running strategy {strategy_name}: {e}")

if __name__ == "__main__":
    main()