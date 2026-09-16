import os
import sys
import shutil
import argparse
import logging
import random
import time
import tiktoken
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


# Silence ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    """
    Decoupled Retriever Wrapper:
    Fetches deep retrieval pool (k=64) for Mismatch@K evaluation,
    while passing only top-k (e.g., 5) to the LLM and RAGAS evaluator.
    """
    def __init__(self, engine: RetrievalEngine, strategy_name: str, generation_top_k: int = 5):
        self.engine = engine
        self.strategy_name = strategy_name
        self.generation_top_k = generation_top_k
        self.last_retrieved_contexts: List[Any] = []

    def invoke(self, query: str):
        # 1. Fetch full candidate pool (64 chunks) from engine
        all_docs = self.engine.retrieve(query, strategy=self.strategy_name)
        self.last_retrieved_contexts = all_docs
        
        # 2. Slice only top-k for generation and RAGAS evaluation
        return all_docs[:self.generation_top_k]


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
            chunk_size=100,
            max_retries=20,
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
    
    if args.dataset == "cuad":
        corpus_dir = args.corpus_dir or "data"
        persist_dir = args.persist_dir or "chroma_db_cuad"
    else:
        corpus_dir = args.corpus_dir or "data_museums"
        persist_dir = args.persist_dir or "chroma_db_museums"

    logger.info(f"Starting RAG System [{args.dataset.upper()} Dataset | {args.mode.upper()} Mode]...")

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
            llm=llm,
            force_reindex=args.force_reindex,
            use_summary_chunking=(args.dataset == "museums")
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

    # 2. Setup Retrieval Engine with Deep Retrieval (top_k=64 for Mismatch@K)
    retrieval_engine = RetrievalEngine(vectorstore=vectorstore, llm=llm, top_k=64)
    
    # Decouple: pass generation_top_k (e.g., 5) to the wrapper for LLM generation/RAGAS
    generation_k = 5
    strategies = {
        "Dense_Similarity": StrategyRetrieverWrapper(retrieval_engine, "dense", generation_top_k=generation_k),
        "MMR_Search": StrategyRetrieverWrapper(retrieval_engine, "mmr", generation_top_k=generation_k),
        "HyDE_Search": StrategyRetrieverWrapper(retrieval_engine, "hyde", generation_top_k=generation_k),
    }
    generator = RAGGenerator(llm=llm, model_name=model_name)

    # 3. Execution Logic based on Mode
    if args.mode == "benchmark":
        evaluator = Evaluator(llm=llm, embeddings=embeddings, output_dir=args.results_dir)

        test_questions = [
            "What is the governing law specified in the Innoviva Collaboration Agreement?",
            "What are the termination rights and notice periods for breach in the Innoviva contract?",
            "Does the Zogenix Distributor Agreement contain an exclusivity or non-compete clause?",
            "How is 'Confidential Information' defined in the Innoviva agreement?",
            "What is the initial term of the Zogenix agreement?",
            "Is there a provision for automatic renewal in the Innoviva Collaboration Agreement?",
            "What happens to intellectual property developed during the Innoviva collaboration?",
            "Are there any indemnification obligations for third-party claims in the Zogenix contract?",
            "What is the cap on aggregate liability for Innoviva?",
            "Which party bears the risk of loss during product shipment in the Zogenix agreement?",
            "Are there specific payment terms or invoice deadlines for Zogenix?",
            "Is consent required before assigning the Innoviva contract to a third party?",
            "Does the Zogenix contract include a severability clause?",
            "What constitutes a Force Majeure event in the Innoviva Collaboration Agreement?",
            "What is the dispute resolution or arbitration mechanism for Zogenix?",
            "Are there any audit rights granted to the parties in the Innoviva contract?",
            "Do confidentiality obligations survive the termination of the Zogenix agreement?",
            "What are the insurance requirements for the contractor in the Innoviva agreement?",
            "Is there a non-solicitation clause regarding employees in the Zogenix contract?",
            "How are changes of control or acquisitions handled by Innoviva?",
            "What are the warranty provisions for delivered products in the Zogenix Distributor Agreement?",
            "Who owns the pre-existing background intellectual property in the Innoviva collaboration?",
            "What is the required procedure for amending the Zogenix agreement?",
            "Where should official legal notices be sent according to the Innoviva contract?",
            "Does the Zogenix agreement state that the parties are independent contractors?"
        ]
        
        ground_truths = [
            "The agreement is governed by the laws of the State of Delaware.",
            "Either party may terminate upon 30 days written notice of a material breach.",
            "Yes, Section 4 contains an exclusive distribution rights clause.",
            "Confidential Information includes any non-public business, financial, or technical data marked as confidential.",
            "The initial term is typically set for three to five years from the Effective Date.",
            "Yes, the agreement automatically renews for successive one-year periods unless written notice is given.",
            "Intellectual property jointly developed shall be jointly owned, while pre-existing IP remains with the original owner.",
            "The Supplier agrees to indemnify and hold harmless the Buyer against any third-party IP infringement claims.",
            "Liability is generally capped at the total fees paid by the Customer in the twelve months preceding the claim.",
            "Risk of loss passes to the Buyer upon delivery to the common carrier.",
            "Invoices are due and payable net 30 to 45 days from the date of receipt.",
            "Neither party may assign this Agreement without the prior written consent of the other party.",
            "If any provision is held invalid, the remaining provisions shall continue in full force and effect.",
            "Force Majeure includes Acts of God, natural disasters, war, terrorism, and labor strikes.",
            "Disputes shall be resolved by binding arbitration under standard commercial rules.",
            "A party may audit the other's records upon 15 to 30 days prior written notice.",
            "Confidentiality obligations survive for a period of three to five years after contract termination.",
            "Commercial General Liability insurance must be maintained with specified limits per occurrence.",
            "The parties agree not to solicit each other's employees for a period of 12 months.",
            "Upon a Change of Control, the other party generally retains the right to terminate the agreement.",
            "The Seller warrants the goods will be free from defects in material and workmanship for a stated period.",
            "Each party retains exclusive ownership of its pre-existing Background IP.",
            "The agreement may only be amended by a written instrument signed by authorized representatives of both parties.",
            "Notices must be sent via certified mail or recognized overnight courier to the addresses listed in the signature block.",
            "Yes, the agreement explicitly states that it does not create a partnership, agency, or employer-employee relationship."
        ]

        ground_truth_docs = [
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt",
            "INNOVIVA_INC_08_07_2014-EX-10.1-COLLABORATION_AGREEMENT.txt",
            "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor_Agreement.txt"
        ]
        
        strategy_trial_outputs: Dict[str, List[List[Dict[str, Any]]]] = {s: [] for s in strategies}
        k_values = [1, 2, 4, 8, 16, 32, 64]

        for trial_idx in range(config.num_trials):
            logger.info(f"=== Starting Trial Execution {trial_idx + 1}/{config.num_trials} ===")
            strategy_items = list(strategies.items())
            random.shuffle(strategy_items)

            for strategy_name, retriever in strategy_items:
                single_trial_run = []
                for question, gt_answer, gt_doc in zip(test_questions, ground_truths, ground_truth_docs):
                    t_start = time.time()
                    
                    # 1. Execute generation (receives top 5 chunks via wrapper)
                    run_output = generator.run_pipeline(retriever, question)
                    
                    # 2. Grab full 64 candidate chunks from wrapper
                    deep_candidate_pool = retriever.last_retrieved_contexts
                    
                    # 3. Extract filename strings from metadata (JSON-safe, no raw Document objects)
                    candidate_doc_names = [
                        Path(str(d.metadata.get("source_file") or d.metadata.get("source") or "")).name
                        for d in deep_candidate_pool
                    ]
                    
                    # 4. Calculate Document Mismatch across all k thresholds
                    target_doc_name = Path(gt_doc).name.strip()
                    mismatch_metrics = {}
                    for k in k_values:
                        top_k_names = candidate_doc_names[:k]
                        found = any(target_doc_name == name or target_doc_name in name for name in top_k_names)
                        mismatch_metrics[f"mismatch@k={k}"] = 0.0 if found else 1.0
                    
                    # 5. Compute operational & lexical metrics on top-5 contexts
                    op_lex_metrics = evaluator.compute_operational_and_lexical_metrics(
                        query=question,
                        retrieved_contexts=run_output.get("contexts", []),
                        generated_answer=run_output.get("answer", ""),
                        ground_truth_answer=gt_answer,
                        start_time=t_start,
                        model_name=config.llm_model
                    )
                    
                    if "metrics" not in run_output:
                        run_output["metrics"] = {}
                    run_output["metrics"].update(mismatch_metrics)
                    run_output["metrics"].update(op_lex_metrics)
                    
                    single_trial_run.append(run_output)
                    
                strategy_trial_outputs[strategy_name].append(single_trial_run)

        final_strategy_results = []
        for strategy_name in strategies:
            trial_data = strategy_trial_outputs[strategy_name]
            eval_summary = evaluator.evaluate_multi_trial_strategy(
                strategy_name=strategy_name,
                trial_outputs=trial_data,
                ground_truths=ground_truths,
                ground_truth_docs=ground_truth_docs # <--- Pass this to calculate Mismatch@K
            )
            final_strategy_results.append(eval_summary)

        config_manifest = {
            "llm_model": model_name,
            "embedding_model": embedding_name,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "retrieval_depth_k": 64,
            "generation_context_k": generation_k,
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
