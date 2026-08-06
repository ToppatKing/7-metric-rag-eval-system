import os
import time
import logging
import pandas as pd
from langchain_openai import ChatOpenAI

from src.config import RAGConfig
from src.indexer import DocumentIndexer
from src.retriever import RetrievalEngine
from src.generator import RAGGenerator
from src.evaluator import Evaluator

# Suppress noisy library logs
logging.basicConfig(level=logging.WARNING)

def prompt_for_valid_folder() -> str:
    """Interactively prompts the user for a valid directory path."""
    while True:
        folder_path = input("\n[Step 1/3] Enter the path to your document folder:\n> ").strip()
        
        # Remove quotation marks if the user drag-and-dropped a folder into the terminal
        folder_path = folder_path.strip("'\"")
        
        if not os.path.exists(folder_path):
            print(f"❌ Error: Path '{folder_path}' does not exist. Please try again.")
            continue
            
        if not os.path.isdir(folder_path):
            print(f"❌ Error: Path '{folder_path}' is a file, not a folder. Please enter a folder path.")
            continue
            
        # Check if the folder contains .pdf or .txt files
        matching_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.pdf', '.txt'))
        ]
        
        if not matching_files:
            print(f"⚠️ Warning: No .pdf or .txt files found in '{folder_path}'.")
            retry = input("Do you want to choose a different folder? (y/n): ").strip().lower()
            if retry == 'y':
                continue
                
        return folder_path

def main():
    print("="*60)
    print(" Domain-Specific RAG System & Rigorous Evaluator ")
    print("="*60)
    
    # Initialize Core LLM Backbone
    config = RAGConfig()
    llm = ChatOpenAI(model=config.llm_model, api_key=config.openai_api_key, temperature=0.0)
    
    # 1. Ask User for Document Set Path
    data_dir = prompt_for_valid_folder()
    
    print("\nProcessing Documents & Initializing Vector Index...")
    indexer = DocumentIndexer(config)
    try:
        vectorstore = indexer.build_or_load_index(data_dir=data_dir)
    except Exception as e:
        print(f"❌ Failed to process documents: {e}")
        return

    # 2. Ask User for Question and Ground Truth
    print("\n[Step 2/3] Enter Query and Reference Answer")
    question = input("Enter your Question:\n> ").strip()
    while not question:
        question = input("Question cannot be empty. Enter your Question:\n> ").strip()
        
    ground_truth = input("\nEnter Ground Truth Answer (Reference answer for evaluation):\n> ").strip()
    while not ground_truth:
        ground_truth = input("Ground Truth cannot be empty. Enter Ground Truth Answer:\n> ").strip()

    # 3. Initialize Retrieval & Evaluation Components
    retriever = RetrievalEngine(vectorstore, llm, top_k=config.top_k)
    generator = RAGGenerator(llm, model_name=config.llm_model)
    evaluator = Evaluator()

    strategies = ["dense", "mmr", "hyde"]
    results_list = []

    print("\n[Step 3/3] Running Simulations Across Strategies...")
    for strategy in strategies:
        print(f"\n" + "-"*40)
        print(f" Running Strategy: {strategy.upper()} ")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            # Step A: Retrieve Context
            retrieved_docs = retriever.retrieve(query=question, strategy=strategy)
            contexts_str_list = [doc.page_content for doc in retrieved_docs]
            
            # Step B: Generate Answer
            gen_output = generator.generate(question=question, context_docs=retrieved_docs)
            answer = gen_output["answer"]
            
            total_latency = time.time() - start_time
            
            print(f"\nGenerated Answer ({strategy.upper()}):\n{answer}\n")
            
            # Step C: Evaluate
            print("Evaluating metrics (Faithfulness, Relevancy, Precision, Recall, ROUGE-L)...")
            metrics = evaluator.run_evaluation(
                question=question,
                answer=answer,
                contexts=contexts_str_list,
                ground_truth=ground_truth,
                latency=total_latency,
                token_efficiency=gen_output["token_efficiency"]
            )
            
            result_row = {"Strategy": strategy.upper()}
            result_row.update(metrics)
            results_list.append(result_row)
            
        except Exception as e:
            print(f"❌ Error during {strategy.upper()} simulation: {e}")

    # 4. Final Output Display
    print("\n\n" + "="*60)
    print(" EVALUATION RESULTS COMPARISON ")
    print("="*60)
    
    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results.set_index("Strategy", inplace=True)
        
        # Display transpose view for easy scanning
        print(df_results.T.to_markdown(floatfmt=".4f"))
        
        df_results.to_csv("rag_evaluation_results.csv")
        print("\n✅ Results successfully exported to 'rag_evaluation_results.csv'")

if __name__ == "__main__":
    main()
