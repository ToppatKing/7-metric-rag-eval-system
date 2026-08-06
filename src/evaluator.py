import math
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class Evaluator:
    """
    RAG Evaluator that explicitly accepts LLM/Embedding dependencies, 
    safeguards against masked failures, and generates atomic, timestamped artifacts.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        output_dir: str | Path = "results",
    ):
        # 1. P2 Fix: Explicitly inject evaluator dependencies
        self.llm = llm
        self.embeddings = embeddings
        self.output_dir = Path(output_dir)
        
        self.metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]

    def _validate_scores(self, result_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        P1 Fix: Validate that required scores are finite and not artificially zeroed by errors.
        """
        for metric in self.metrics:
            score = result_dict.get(metric.name)
            
            if score is None:
                return False, f"Metric '{metric.name}' is missing from the results."
                
            try:
                # Catch NaN or Inf values disguised as floats
                if not math.isfinite(float(score)):
                    return False, f"Metric '{metric.name}' returned a non-finite score: {score}"
            except (ValueError, TypeError):
                return False, f"Metric '{metric.name}' returned a non-numeric score: {score}"
                
        return True, ""

    def evaluate_strategy(
        self,
        strategy_name: str,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[List[str]],
        retrieved_doc_ids: List[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single RAG strategy and captures complete diagnostics.
        """
        logger.info(f"Evaluating strategy: {strategy_name}")
        
        # Prepare HuggingFace dataset format required by Ragas 0.1.7
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths, 
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        status = "SUCCESS"
        error_message = None
        metrics_scores = {}
        detailed_rows = []

        try:
            # 2. P1 Fix: Set raise_exceptions=True so API/Judge failures are not masked as 0.0
            result = evaluate(
                dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=True, 
            )
            
            metrics_scores = dict(result)
            
            # 3. P1 Fix: Additional validation to catch any hidden NaNs
            is_valid, validation_err = self._validate_scores(metrics_scores)
            if not is_valid:
                status = "FAILED"
                error_message = validation_err

        except Exception as e:
            logger.error(f"Execution failure during evaluation of {strategy_name}: {e}")
            # 4. P1 Fix: Mark the strategy as failed rather than faking a numeric score
            status = "FAILED"
            error_message = str(e)
            metrics_scores = {metric.name: None for metric in self.metrics}
        
        # 5. P1 Fix: Assemble detailed per-question records for auditability
        for idx in range(len(questions)):
            row = {
                "question": questions[idx],
                "expected_answer": ground_truths[idx] if ground_truths else [],
                "generated_answer": answers[idx],
                "retrieved_contexts": contexts[idx],
                "retrieved_doc_ids": retrieved_doc_ids[idx] if retrieved_doc_ids else [],
            }
            detailed_rows.append(row)
            
        return {
            "strategy_name": strategy_name,
            "status": status,
            "error": error_message,
            "scores": metrics_scores,
            "details": detailed_rows,
        }
        
    def write_artifacts(
        self,
        run_name: str,
        experiment_config: Dict[str, Any],
        strategy_results: List[Dict[str, Any]]
    ) -> Path:
        """
        P1 Fix: Writes timestamped, auditable, and immutable artifacts for the run.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_identifier = f"{run_name}_{timestamp}"
        run_dir = self.output_dir / run_identifier
        
        # 6. P1 Fix: Avoid silently overwriting prior experiments (exist_ok=False)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.error(f"Collision: Artifact directory {run_dir} already exists.")
            run_identifier = f"{run_identifier}_retry"
            run_dir = self.output_dir / run_identifier
            run_dir.mkdir(parents=True, exist_ok=False)
        
        # Store experiment manifest (provenance, fingerprints)
        manifest_path = run_dir / "run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_identifier": run_identifier,
                "timestamp": timestamp,
                "config": experiment_config,
            }, f, indent=2)
            
        # Store summary CSV for statistical comparison
        summary_rows = []
        for res in strategy_results:
            row = {
                "strategy": res["strategy_name"],
                "status": res["status"],
                "error": res["error"],
            }
            row.update(res["scores"])
            summary_rows.append(row)
            
        summary_df = pd.DataFrame(summary_rows)
        summary_path = run_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Store explicit row-by-row JSON records for deep auditing
        details_path = run_dir / "detailed_results.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(strategy_results, f, indent=2)
            
        logger.info(f"Run artifacts successfully generated and secured at: {run_dir}")
        return run_dir
