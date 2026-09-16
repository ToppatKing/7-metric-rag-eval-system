import os
import json
import glob
from pathlib import Path
import pandas as pd

def get_latest_benchmark_json(results_dir: str = "results") -> Path:
    """Finds the detailed_results.json from the most recent benchmark run directory."""
    benchmark_dirs = sorted(glob.glob(os.path.join(results_dir, "*benchmark*")), key=os.path.getmtime)
    if not benchmark_dirs:
        raise FileNotFoundError(f"No benchmark directories found in '{results_dir}'.")
    
    latest_dir = Path(benchmark_dirs[-1])
    json_path = latest_dir / "detailed_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"'{json_path}' does not exist in the latest run directory.")
    
    return json_path

def generate_academic_report():
    print("Locating most recent benchmark results...")
    try:
        json_path = get_latest_benchmark_json("results")
        print(f"Loading data from: {json_path}")
    except Exception as e:
        print(f"Error finding results: {e}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    k_values = [1, 2, 4, 8, 16, 32, 64]
    
    for strategy in data:
        name = strategy.get("strategy_name", "Unknown Strategy")
        stats = strategy.get("statistics", {})
        
        row_data = {
            "Retrieval Strategy": name.replace("_", " "),
            "Context Precision": round(stats.get("context_precision_median") or 0, 4),
            "Context Recall": round(stats.get("context_recall_median") or 0, 4),
            "Faithfulness": round(stats.get("faithfulness_median") or 0, 4),
            "Answer Relevancy": round(stats.get("answer_relevancy_median") or 0, 4),
            "ROUGE-L F1": round(stats.get("rouge_l_f1_median") or 0, 4),           # <--- Added ROUGE-L
            "Token Efficiency": round(stats.get("token_efficiency_median") or 0, 4), # <--- Added Token Efficiency
            "Avg Latency (s)": round(stats.get("latency_median_sec") or 0, 2),
        }
        
        for k in k_values:
            mismatch_val = stats.get(f"mismatch@k={k}_median", stats.get(f"mismatch@k={k}", 0))
            row_data[f"Mismatch@{k}"] = round(mismatch_val if mismatch_val is not None else 0, 4)
            
        records.append(row_data)
        
    df = pd.DataFrame(records)
    
    print("\n" + "="*120)
    print(" 7-METRIC AGGREGATE RESULTS FOR THESIS (MARKDOWN FORMAT)")
    print("="*120 + "\n")
    print(df.to_markdown(index=False))
    
    output_csv = "results/thesis_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSummary successfully written to: {output_csv}")

if __name__ == "__main__":
    generate_academic_report()
