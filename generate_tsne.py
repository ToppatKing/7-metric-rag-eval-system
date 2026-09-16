import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def generate_tsne_projection(persist_dir: str = "chroma_db_cuad"):
    """Extracts embeddings natively from ChromaDB and plots a 2D t-SNE projection."""
    import chromadb
    print(f"Scanning ChromaDB directory: {persist_dir}...")
    
    # Connect natively to ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()
    
    if not collections:
        print("ERROR: No collections found in this directory. Are you sure this is the right path?")
        return
        
    # Auto-detect the collection that actually contains your documents
    target_collection = None
    for c in collections:
        print(f"Found collection '{c.name}' with {c.count()} chunks.")
        if c.count() > 0:
            target_collection = c
            break
            
    if not target_collection:
        print("ERROR: Collections exist, but they are all empty!")
        return

    print(f"Extracting vectors from collection: {target_collection.name}...")
    # Fetch embeddings and documents directly
    data = target_collection.get(include=['embeddings', 'documents'])
    
    all_embeddings = data['embeddings']
    all_documents = data['documents']
    
    if not all_embeddings:
        print("No embeddings found in the extracted data.")
        return

    # Sample up to 800 chunks to keep the visualization clean and fast
    sample_size = min(800, len(all_embeddings))
    indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
    
    sampled_embeddings = np.array(all_embeddings)[indices]
    sampled_docs = np.array(all_documents)[indices]
    
    print(f"Running t-SNE dimensionality reduction on {sample_size} chunks (1536D -> 2D)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    reduced_embeddings = tsne.fit_transform(sampled_embeddings)
    
    # Separate points based on a keyword to show dense clustering
    x_coords = reduced_embeddings[:, 0]
    y_coords = reduced_embeddings[:, 1]
    
    # Categorize chunks to show how boilerplate clusters together
    labels = ["Termination Clause" if "terminate" in doc.lower() or "termination" in doc.lower() 
              else "General Contract Text" for doc in sampled_docs]

    print("Plotting the vector space...")
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    sns.scatterplot(
        x=x_coords, y=y_coords, 
        hue=labels, 
        palette={"Termination Clause": "#d62728", "General Contract Text": "#1f77b4"},
        alpha=0.6, s=40, edgecolor=None
    )
    
    plt.title("t-SNE Projection of CUAD Vector Space", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Semantic Clusters", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("cuad_tsne_projection.pdf", format="pdf")
    print("Vector projection successfully saved as 'cuad_tsne_projection.pdf'")

if __name__ == "__main__":
    # If your CUAD database is in 'chroma_db', change the string below!
    generate_tsne_projection("chroma_db_cuad")
