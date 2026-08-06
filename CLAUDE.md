# Agent Guide: 7-Metric RAG Evaluation System

This file is the repository briefing for coding agents. Keep changes small,
preserve the existing architecture, and update this guide when behavior or
commands change.

## Project Purpose

This Python CLI compares three retrieval strategies over a PDF/TXT corpus:

- Dense vector similarity
- Maximal Marginal Relevance (MMR)
- Hypothetical Document Embeddings (HyDE)

Each strategy retrieves context, generates an OpenAI-backed answer, and is
evaluated with Ragas, ROUGE-L, latency, and token-efficiency metrics.

## Repository Map

- `main.py`: interactive CLI and end-to-end orchestration
- `src/config.py`: configuration dataclass and `.env` loading
- `src/indexer.py`: document loading, chunking, embeddings, and Chroma storage
- `src/retriever.py`: dense, MMR, and HyDE retrieval
- `src/generator.py`: context-bounded answer generation and token accounting
- `src/evaluator.py`: Ragas and ROUGE-L evaluation
- `requirements.txt`: Python runtime dependencies
- `README.md`: user-facing architecture and evaluation documentation

There is currently no automated test suite, CI workflow, lint configuration,
package metadata, or `LICENSE` file.

## Runtime Flow

1. `RAGConfig` loads `.env` and requires `OPENAI_API_KEY`.
2. The user selects a document directory.
3. `DocumentIndexer` loads or builds a persistent Chroma index.
4. The user enters one question and an optional ground-truth answer.
5. Dense, MMR, and HyDE run sequentially.
6. Each strategy retrieves documents and generates an answer.
7. `Evaluator` computes available metrics.
8. Results print as a transposed table and overwrite
   `rag_evaluation_results.csv`.

Reported latency covers retrieval plus answer generation. It excludes indexing
and metric evaluation.

## Setup and Commands

Use Python 3.10 or newer and run commands from the repository root.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` locally:

```env
OPENAI_API_KEY=your-key
```

Run the application:

```bash
python main.py
```

Minimum offline validation after Python changes:

```bash
python -m compileall -q main.py src
```

An end-to-end run needs a valid OpenAI key, network access, and a PDF/TXT
corpus, and it incurs API usage.

## Configuration Defaults

`src/config.py` defines:

| Setting | Default |
| --- | --- |
| Chat model | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Chroma base path | `./chroma_db` |
| Chunk size | `1000` characters |
| Chunk overlap | `200` characters |
| Retrieved chunks | `5` |

Only `OPENAI_API_KEY` is environment-driven. Other settings require explicit
`RAGConfig` constructor overrides or code changes.

## Retrieval Details

- Dense uses similarity search with `k=top_k`.
- MMR uses `k=top_k`, `fetch_k=top_k * 4`, and `lambda_mult=0.5`.
- HyDE first asks the chat model for a hypothetical factual answer, then uses
  that passage for similarity search. This adds an LLM call and extra latency.
- Strategy names accepted by the router are exact lowercase values: `dense`,
  `mmr`, and `hyde`.

## Evaluation Behavior

Always reported:

- Faithfulness
- Answer Relevancy
- Latency in seconds
- Token Efficiency (`answer_tokens / context_tokens`)

Reported only when a nonblank ground truth is supplied:

- Context Precision
- Context Recall
- ROUGE-L

The three reference-dependent metrics are `NaN` without ground truth. Ragas is
called with `raise_exceptions=False`, so metric failures may appear as missing,
zero, or invalid scores rather than stopping a strategy. Inspect warnings when
debugging unexpected evaluation output.

`RAGGenerator` returns its own `generation_time`, but `main.py` does not report
that field; it reports retrieval-plus-generation latency instead.

## Index and Artifact Hazards

Chroma cache directories are named `chroma_db_<hash>`, where the hash is based
only on the corpus directory's absolute path. The cache key does not include
file contents, chunk settings, or the embedding model.

Consequences:

- Edited, added, or removed documents do not invalidate a populated cache.
- Changed chunk or embedding settings can reuse incompatible stale data.
- A partially built nonempty cache is treated as complete.
- Moving the corpus creates a separate cache.

The indexer recursively matches lowercase `*.pdf` and `*.txt`. Uppercase file
extensions are ignored on Linux. Its CLI precheck only inspects immediate child
files, so it can warn even when supported files exist in nested directories.

Generated artifacts are not ignored because the repository has no `.gitignore`:

- `chroma_db_<hash>/`
- `rag_evaluation_results.csv`
- `__pycache__/` directories

Do not commit generated indexes, API keys, `.env`, or evaluation output unless
the user explicitly requests it.

## Engineering Guidance

- Follow the existing class boundaries: configuration, indexing, retrieval,
  generation, and evaluation are intentionally separate.
- Prefer focused fixes over introducing new frameworks or abstractions.
- Preserve the three-strategy comparison unless the requested behavior changes
  it explicitly.
- Keep prompts deterministic unless experimentation is the task; the shared
  chat model currently uses `temperature=0.0`.
- Treat token efficiency as a compression proxy, not a higher-is-better score.
- When changing cache semantics, test stale, partial, and moved-corpus cases.
- When changing metrics, test both reference-based and reference-free paths.
- Do not silently broaden supported formats; add the loader, recursive discovery,
  CLI validation, documentation, and tests together.
- Avoid relying on `vectorstore._collection`; it is a private Chroma attribute
  already used by the current implementation and should be replaced carefully.

## Known Documentation and Dependency Notes

- README metric targets and example scores are illustrative, not enforced.
- README describes latency slightly more broadly than the implementation.
- README advertises an MIT license, but no license file exists.
- `pandas` is unpinned.
- `datasets`, `langchain-core`, `langchain-text-splitters`, and `tabulate` are
  used directly or needed at runtime but currently arrive transitively.