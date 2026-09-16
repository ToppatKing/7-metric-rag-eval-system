import os
from pathlib import Path
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns

def count_tokens_in_corpus(corpus_dir: str = "data"):
    """Reads all text files in the corpus and counts their tokens."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_counts = []
    
    corpus_path = Path(corpus_dir)
    files = list(corpus_path.rglob("*.txt"))
    
    print(f"Analyzing {len(files)} files in {corpus_dir}...")
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                token_counts.append(len(tokenizer.encode(text)))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return token_counts

def plot_heavy_tail_histogram(token_counts):
    """Generates an academic-grade histogram of token distributions."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot the histogram with a Kernel Density Estimate (KDE) line
    sns.histplot(token_counts, bins=50, kde=True, color="#1f77b4", edgecolor="black")
    
    # Add an absolute vertical line showing a standard 8K context limit
    plt.axvline(x=8192, color='red', linestyle='--', linewidth=2, label='Standard 8K Context Limit')
    
    plt.title("CUAD Corpus Token Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Document Length (Tokens)", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency (Number of Contracts)", fontsize=12, fontweight="bold")
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("cuad_token_distribution.pdf", format="pdf")
    print("Histogram successfully saved as 'cuad_token_distribution.pdf'")

if __name__ == "__main__":
    # Point this to the directory where your CUAD .txt files are stored
    counts = count_tokens_in_corpus("data")
    if counts:
        plot_heavy_tail_histogram(counts)
