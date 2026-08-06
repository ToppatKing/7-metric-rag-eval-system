# 🔬 Domain-Specific RAG System with Rigorous Evaluation

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.16-green)
![Ragas](https://img.shields.io/badge/Ragas-0.1.7-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Project Overview
As Retrieval-Augmented Generation (RAG) becomes the standard for domain-specific AI, assessing these systems via "vibes" or manual review is no longer sufficient. This repository provides a **fully automated, rigorous evaluation framework** that transitions RAG development from a "ship-and-pray" loop to a "measure-and-improve" loop.

This project implements a multi-strategy RAG pipeline—comparing **Dense Vector**, **Maximal Marginal Relevance (MMR)**, and **Hypothetical Document Embeddings (HyDE)**—and evaluates them simultaneously against a robust 7-metric framework. It isolates retrieval failures from generation failures, utilizing an LLM-as-a-judge paradigm (via `ragas`) alongside deterministic n-gram overlap metrics (ROUGE-L) and system heuristics (Latency & Token Efficiency).

**Designed for academic and production research, this repository serves to demonstrate advanced competency in modern LLM orchestration, vector search optimization, and reproducible AI evaluation.**

---

## 🏗️ System Architecture

The pipeline is split into three main components: Document Ingestion, Multi-Strategy Retrieval, and the 7-Metric Evaluation Engine.

```mermaid
graph TD
    %% Styling
    classDef ingestion fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef retrieval fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef generation fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef evaluation fill:#fce4ec,stroke:#c2185b,stroke-width:2px;

    %% Ingestion Phase
    A[Domain Documents<br/>PDF / TXT] --> B(Text Splitter & Chunker)
    B --> C[(ChromaDB<br/>Vector Store)]
    class A,B,C ingestion;

    %% Retrieval Phase
    D[User Query] --> E{Retrieval Router}
    C -.-> F[Dense Vector]
    C -.-> G[MMR]
    C -.-> H[HyDE]
    E --> F
    E --> G
    E --> H
    F & G & H --> I[Retrieved Context Chunks]
    class D,E,F,G,H,I retrieval;

    %% Generation Phase
    I --> J(Generator LLM<br/>gpt-4o-mini)
    D --> J
    J --> K[Final Answer]
    class J,K generation;

    %% Evaluation Phase
    I -.-> L((Evaluation Engine))
    K -.-> L
    D -.-> L
    M[Ground Truth] -.-> L
    L --> N[7-Metric Output DataFrame]
    class L,M,N evaluation;
