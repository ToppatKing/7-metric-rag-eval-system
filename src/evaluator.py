import math
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
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
    RAG Evaluator that supports multi-trial statistical controls, 
    safeguards against masked failures, and generates timestamped artifacts.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        output_dir: str | Path = "results",
    ):
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
        """Validate that required scores are finite and not zeroed by errors."""
        for metric in self.metrics:
            score = result_dict.get(metric.name)
            if score is None:
                return False, f"Metric '{metric.name}' is missing."
            try:
                if not math.isfinite(float(score)):
                    return False, f"Metric '{metric.name}' returned non-finite score: {score}"
            except (ValueError, TypeError):
                return False, f"Metric '{metric.name}' returned non-numeric score: {score}"
        return True, ""

    def evaluate_multi_trial_strategy(
        self,
        strategy_name: str,
        trial_outputs: List[List[Dict[str, Any]]],
        ground_truths: List[List[str]],
    ) -> Dict[str, Any]:
        """
        P2 Fix: Evaluates a strategy across N trials and calculates statistical metrics
        (median, standard deviation, and p95 latency).
        """
        logger.info(f"Evaluating {len(trial_outputs)} trial(s) for strategy: {strategy_name}")
        
        trial_score_lists: Dict[str, List[float]] = {m.name: [] for m in self.metrics}
        trial_latencies: List[float] = []
        trial_costs: List[float] = []
        trial_compressions: List[float] = []

        raw_trial_records = []
        overall_status = "SUCCESS"
        last_error = None

        for trial_idx, single_trial_run in enumerate(trial_outputs):
            questions = [item["question"] for item in single_trial_run]
            answers = [item["answer"] for item in single_trial_run]
            contexts = [item["contexts"] for item in single_trial_run]
            
            # Aggregate token & timing stats for this trial
            trial_total_latency = sum(item["metrics"]["total_latency_sec"] for item in single_trial_run)
            trial_total_cost = sum(item["metrics"]["estimated_cost_usd"] for item in single_trial_run)
            avg_compression = np.mean([item["metrics"]["context_to_answer_compression_ratio"] for item in single_trial_run])

            trial_latencies.append(trial_total_latency)
            trial_costs.append(trial_total_cost)
            trial_compressions.append(avg_compression)

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truths,
            })

            try:
                result = evaluate(
                    dataset,
                    metrics=self.metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=True,
                )
                scores = dict(result)
                is_valid, err = self._validate_scores(scores)
                
                if is_valid:
                    for m in self.metrics:
                        trial_score_lists[m.name].append(float(scores[m.name]))
                else:
                    overall_status = "PARTIAL_FAILURE"
                    last_error = err

            except Exception as e:
                logger.error(f"Trial {trial_idx + 1} failed for {strategy_name}: {e}")
                overall_status = "PARTIAL_FAILURE"
                last_error = str(e)

            raw_trial_records.append({
                "trial_index": trial_idx + 1,
                "questions_detail": single_trial_run
            })

        # Calculate statistical dispersion across trials
        stats_summary = {}
        for metric_name, values in trial_score_lists.items():
            if values:
                stats_summary[f"{metric_name}_median"] = round(float(np.median(values)), 4)
                stats_summary[f"{metric_name}_std"] = round(float(np.std(values)), 4)
            else:
                stats_summary[f"{metric_name}_median"] = None
                stats_summary[f"{metric_name}_std"] = None

        # Calculate timing/efficiency statistics
        stats_summary["latency_p95_sec"] = round(float(np.percentile(trial_latencies, 95)), 4) if trial_latencies else None
        stats_summary["latency_median_sec"] = round(float(np.median(trial_latencies)), 4) if trial_latencies else None
        stats_summary["avg_compression_ratio"] = round(float(np.mean(trial_compressions)), 4) if trial_compressions else None
        stats_summary["avg_cost_usd"] = round(float(np.mean(trial_costs)), 6) if trial_costs else None

        return {
            "strategy_name": strategy_name,
            "status": overall_status if trial_score_lists[self.metrics[0].name] else "FAILED",
            "error": last_error,
            "statistics": stats_summary,
            "trials_raw": raw_trial_records,
        }

    def write_artifacts(
        self,
        run_name: str,
        experiment_config: Dict[str, Any],
        strategy_results: List[Dict[str, Any]]
    ) -> Path:
        """Writes timestamped, auditable, and immutable artifacts."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_identifier = f"{run_name}_{timestamp}"
        run_dir = self.output_dir / run_identifier
        
        run_dir.mkdir(parents=True, exist_ok=False)

        # Store experiment manifest
        manifest_path = run_dir / "run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_identifier": run_identifier,
                "timestamp": timestamp,
                "config": experiment_config,
            }, f, indent=2)

        # Store statistical summary CSV
        summary_rows = []
        for res in strategy_results:
            row = {
                "strategy": res["strategy_name"],
                "status": res["status"],
                "error": res["error"],
            }
            row.update(res["statistics"])
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = run_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Store detailed trial records
        details_path = run_dir / "detailed_results.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(strategy_results, f, indent=2)

        logger.info(f"Run artifacts successfully saved to: {run_dir}")
        return run_dir
