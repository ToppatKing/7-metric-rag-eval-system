# Code Review: 7-Metric RAG Evaluation System

Review date: 2026-08-06

## Executive Summary

The codebase is compact and its major responsibilities are separated cleanly,
but it is not yet suitable for evidence-generating research or production use.
The largest risks are silent reuse of stale or partial indexes, ambiguous metric
failures, missing experiment provenance, and absent protection for credentials
and generated corpus artifacts.

No syntax or installed-dependency conflict was found. The main concerns are
evaluation validity, data integrity, security hygiene, and reproducibility.

## Findings

### P1: Index caches can silently represent stale or partial corpora

Evidence: `src/indexer.py:34-46`, `src/indexer.py:91`

The cache name is derived only from the corpus directory's absolute path. Any
nonempty Chroma collection is treated as complete. The cache identity excludes:

- Document contents, additions, and deletions
- Chunk size and overlap
- Embedding model
- Index schema or application version
- Successful completion of every batch

An interrupted build can leave a partial collection that is accepted as
complete on the next run. Document or configuration changes can similarly reuse
an incompatible index. Every downstream retrieval and evaluation score can then
describe the wrong corpus.

Recommended remediation:

1. Build a manifest containing content hashes, relevant configuration, model,
   schema version, and discovered file list.
2. Build into a temporary collection or directory.
3. Write a completion marker only after every expected file and chunk succeeds.
4. Atomically promote the completed index.
5. Add an explicit force-reindex option.
6. Replace reliance on Chroma's private `_collection` attribute.

### P1: Credentials and sensitive generated artifacts are not protected

Evidence: `README.md:158-162`, `src/indexer.py:35`, `main.py:137`

The setup instructions create an OpenAI credential file in the repository.
Chroma stores source-document chunks locally, and evaluation results are also
written into the repository. No ignore configuration exists. A routine
`git add .` can therefore stage credentials, private corpus content, result
files, and Python bytecode.

Recommended remediation:

- Add ignore rules for credentials, Chroma directories, result files, virtual
  environments, and Python caches.
- Prefer an operating-system cache/data directory outside the repository for
  persisted indexes.
- Document the data-retention and deletion policy for indexed corpora.
- Add secret scanning in CI before accepting external contributions.

### P1: Metric execution failures are indistinguishable from quality scores

Evidence: `src/evaluator.py:49-61`, `main.py:130-138`

Ragas runs with `raise_exceptions=False`. Missing result keys default to `0.0`,
and returned values are not checked for finiteness. Rate limits, judge failures,
or parsing failures can appear as zero or `NaN`, while the CLI still announces
a successful export. Users cannot reliably distinguish a bad system from a
failed evaluator.

Recommended remediation:

- Capture a status and error for every metric.
- Validate required scores with `math.isfinite`.
- Mark the strategy and run incomplete when required metrics fail.
- Never substitute an execution failure with a numeric quality score.
- Preserve evaluator diagnostics in the output artifact.

### P1: Result artifacts are not auditable or reproducible

Evidence: `main.py:118-138`

The CSV contains only strategy names and aggregate scores and overwrites a fixed
filename. It omits the question, reference answer, generated answers, retrieved
contexts, corpus fingerprint, model and chunk configuration, judge model,
dependency versions, timestamps, and failure status. If one strategy fails, the
remaining subset is still exported as a successful comparison.

Recommended remediation:

- Assign a unique run identifier and write timestamped artifacts.
- Store an experiment manifest with corpus, model, prompt, and configuration
  fingerprints.
- Store one explicit row or record per requested strategy, including failures.
- Retain generated answers and retrieved document identifiers for auditability.
- Write results atomically and avoid silently overwriting prior experiments.

### P2: Evaluator model and embedding configuration are implicit

Evidence: `src/evaluator.py:49`, `src/config.py:13-24`

The application does not pass an LLM or embedding implementation to Ragas.
Installed-library defaults therefore select the evaluation judge independently
of `RAGConfig`, and the selected judge is not recorded in results. Evaluation
behavior can drift without an intentional application change.

Recommended remediation:

- Add explicit judge model and judge embedding settings.
- Inject those dependencies into `Evaluator`.
- Record model names, versions, temperatures, and relevant prompts per run.

### P2: Strategy comparisons are not statistically controlled

Evidence: `main.py:83-103`

Each strategy runs once in a fixed dense, MMR, HyDE order. Network variation,
API warm-up, throttling, and stochastic LLM judging can dominate small measured
differences. Temperature zero does not guarantee deterministic API output.

Recommended remediation:

- Run repeated trials and report median, p95, and dispersion.
- Randomize or counterbalance strategy order.
- Separate retrieval, hypothetical-generation, answer-generation, and
  evaluation timings.
- Record failed and retried API calls.

### P2: Token efficiency is a compression ratio, not an efficiency metric

Evidence: `src/generator.py:36-48`, `README.md:118-123`

The metric is `answer_tokens / context_tokens`. It improves when the system
retrieves more irrelevant context or returns a short refusal. It does not
measure answer correctness, information retained, latency, or monetary cost.
Optimizing it independently can degrade system quality.

Recommended remediation:

- Rename it to context-to-answer compression ratio.
- Report absolute input/output tokens and estimated cost.
- Interpret compression only alongside faithfulness, relevancy, and retrieval
  quality.
- Do not enforce a standalone target range without empirical validation.

### P2: Source-document loss is accepted as successful indexing

Evidence: `src/indexer.py:67-96`

Loader failures are logged and skipped, but the index is still declared
successful and cached. Empty text, image-only PDFs, OCR failures, and malformed
encodings are not measured. A materially incomplete corpus can therefore become
the permanent cache for that path.

Recommended remediation:

- Report discovered, loaded, failed, empty, and chunked counts.
- Persist per-file status in the index manifest.
- Reject an empty corpus and define an acceptable failure threshold.
- Add OCR or explicit rejection for image-only PDFs.

### P3: CLI validation and index discovery disagree

Evidence: `main.py:29-35`, `src/indexer.py:53-54`

The CLI checks only immediate children and compares extensions
case-insensitively. The indexer searches recursively but matches lowercase glob
patterns on a case-sensitive filesystem. Nested files can trigger false
warnings, while uppercase extensions are ignored during indexing.

Recommended remediation: implement one shared recursive discovery function and
use it for validation and indexing.

## Testing and Delivery Gaps

No automated tests, CI workflow, lint configuration, dependency lock, or ignore
configuration exists. The minimum high-value test matrix should cover:

- Fresh, reused, stale, partial, and force-rebuilt indexes
- Added, changed, deleted, failed, empty, and uppercase-extension documents
- Dense, MMR, and HyDE routing with deterministic fakes
- Reference-based and reference-free metric assembly
- Ragas exceptions, `NaN`, timeouts, and partial strategy failures
- Result manifest completeness and non-overwriting behavior
- Credential and generated-artifact exclusion

## Recommended Delivery Order

1. Add artifact and credential protection.
2. Make index construction transactional and content-aware.
3. Make metric and strategy failures explicit.
4. Introduce versioned, auditable run artifacts.
5. Inject evaluator dependencies explicitly.
6. Add focused tests and CI.
7. Improve benchmark methodology and metric naming.

## Validation Performed

- `python -m compileall -q main.py src`: passed
- Runtime imports for pandas, Chroma, Ragas, LangChain, and `main`: passed
- `python -m pip check`: passed
- Installed versions observed during review:
  - pandas 3.0.5
  - chromadb 0.4.24
  - ragas 0.1.7
  - langchain 0.1.16

No end-to-end evaluation was run because it requires a representative corpus,
network access, a valid OpenAI credential, and billable API calls.

## Review Assumptions

- The repository's claims of rigorous, reproducible academic and production
  research define the expected engineering standard.
- Corpora may change and may contain confidential or untrusted material.
- Evaluation outputs may influence model, retrieval, or deployment decisions.

If these assumptions are intentionally narrower, some severity levels may be
reduced, but the underlying failure modes remain.