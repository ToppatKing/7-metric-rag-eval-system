import time
import logging
import pandas as pd
from langchain_openai import ChatOpenAI

# Ensure you have an __init__.py in your src/ directory
from src.config import RAGConfig
from src.indexer import DocumentIndexer
from src.retriever import RetrievalEngine
from src.generator import RAGGenerator
from src.evaluator import Evaluator

# Configure logging to prevent messy console output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    print("="*60)
    print(" Domain-Specific RAG System & Rigorous Evaluator ")
    print("="*60)
    
    # 1. Initialize Configuration and Backbone
    config = RAGConfig()
    # Using temperature=0.0 ensures highly deterministic evaluation outputs
    llm = ChatOpenAI(model=config.llm_model, api_key=config.openai_api_key, temperature=0.0)
    
    # 2. Indexing Phase (Ensure your PDFs/TXTs are in data/)
    print("\n[1/3] Initializing Vector Database...")
    indexer = DocumentIndexer(config)
    vectorstore = indexer.build_or_load_index(data_dir="data/")
    
    # 3. Initialize Engine Components
    retriever = RetrievalEngine(vectorstore, llm, top_k=config.top_k)
    generator = RAGGenerator(llm, model_name=config.llm_model)
    evaluator = Evaluator()

    # 4. Interactive Query Intake
    print("\n[2/3] Awaiting Input")
    question = input("Enter your Question:\n> ").strip()
    ground_truth = input("\nEnter the Ground Truth Answer (for Eval):\n> ").strip()
    
    if not question or not ground_truth:
        print("Error: Question and Ground Truth are required. Exiting.")
        return

    strategies = ["dense", "mmr", "hyde"]
    results_list = []

    print("\n[3/3] Running Simulations...")
    for strategy in strategies:
        print(f"\n>> Simulating Strategy: {strategy.upper()} <<")
        
        try:
            # Measure Total System Latency
            start_time = time.time()
            
            # Step A: Retrieve
            retrieved_docs = retriever.retrieve(query=question, strategy=strategy)
            contexts_str_list = [doc.page_content for doc in retrieved_docs]
            
            # Step B: Generate
            gen_output = generator.generate(question=question, context_docs=retrieved_docs)
            answer = gen_output["answer"]
            
            # Finalize Latency
            total_latency = time.time() - start_time
            
            print(f"Generated Answer:\n{answer}\n")
            
            # Step C: Evaluate
            print("Running RAGAS & ROUGE Evaluation... (This may take a moment)")
            metrics = evaluator.run_evaluation(
                question=question,
                answer=answer,
                contexts=contexts_str_list,
                ground_truth=ground_truth,
                latency=total_latency,
                token_efficiency=gen_output["token_efficiency"]
            )
            
            # Store results for the final dataframe
            result_row = {"Strategy": strategy.upper()}
            result_row.update(metrics)
            results_list.append(result_row)
            
        except Exception as e:
            print(f"Error during {strategy.upper()} simulation: {e}")

    # 5. Output Results
    print("\n\n" + "="*60)
    print(" FINAL EVALUATION REPORT ")
    print("="*60)
    
    df_results = pd.DataFrame(results_list)
    df_results.set_index("Strategy", inplace=True)
    
    # Transposing makes the 7 metrics easier to read vertically in the console
    print(df_results.T.to_markdown(floatfmt=".4f"))
    
    # Export to CSV for your thesis appendix
    df_results.to_csv("rag_evaluation_results.csv")
    print("\n[Success] Results exported to 'rag_evaluation_results.csv'")

if __name__ == "__main__":
    main()
