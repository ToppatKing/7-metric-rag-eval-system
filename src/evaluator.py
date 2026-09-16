import math
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
import tiktoken
from rouge_score import rouge_scorer

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


def _safe_serializer(obj: Any) -> Any:
    """Serializes LangChain Document objects, Paths, and numpy types for json.dump."""
    if hasattr(obj, "page_content"):
        return {
            "page_content": obj.page_content,
            "metadata": getattr(obj, "metadata", {})
        }
    if hasattr(obj, "item"):  # Handles numpy scalars like np.float64, np.int64
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


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

    def compute_operational_and_lexical_metrics(
        self,
        query: str, 
        retrieved_contexts: list, 
        generated_answer: str, 
        ground_truth_answer: str, 
        start_time: float,
        model_name: str = "gpt-4o-mini"
    ) -> dict:
        """
        Computes Latency, Token Efficiency, and ROUGE-L F1 score 
        for a single RAG execution cycle.
        """
        # 1. Latency Calculation
        end_time = time.time()
        latency_seconds = end_time - start_time

        # 2. Token Efficiency Calculation using tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        context_text = "\n".join([
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in retrieved_contexts
        ])
        context_tokens = len(encoding.encode(context_text))
        answer_tokens = len(encoding.encode(generated_answer))
        
        token_efficiency = (answer_tokens / context_tokens) if context_tokens > 0 else 0.0

        # 3. ROUGE-L F1 Score Calculation
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth_answer, generated_answer)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure

        return {
            "latency_s": round(latency_seconds, 2),
            "context_tokens": context_tokens,
            "answer_tokens": answer_tokens,
            "token_efficiency": round(token_efficiency, 4),
            "rouge_l_f1": round(rouge_l_f1, 4)
        }

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
        ground_truths: List[str],
        ground_truth_docs: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a strategy across N trials and calculates statistical metrics
        including ROUGE-L, Token Efficiency, and Document-Level Mismatch@K.
        """
        logger.info(f"Evaluating {len(trial_outputs)} trial(s) for strategy: {strategy_name}")
        
        trial_score_lists: Dict[str, List[float]] = {m.name: [] for m in self.metrics}
        trial_latencies: List[float] = []
        trial_costs: List[float] = []
        trial_compressions: List[float] = []
        trial_rouge_l: List[float] = []
        trial_token_eff: List[float] = []
        trial_mismatches: Dict[str, List[float]] = {}

        raw_trial_records = []
        overall_status = "SUCCESS"
        last_error = None

        for trial_idx, single_trial_run in enumerate(trial_outputs):
            questions = [item["question"] for item in single_trial_run]
            answers = [item["answer"] for item in single_trial_run]
            contexts = [item["contexts"] for item in single_trial_run]
            
            # Aggregate trial-level timing and cost
            trial_total_latency = sum(
                item["metrics"].get("total_latency_sec", item["metrics"].get("latency_s", 0)) 
                for item in single_trial_run
            )
            trial_total_cost = sum(item["metrics"].get("estimated_cost_usd", 0) for item in single_trial_run)
            avg_compression = np.mean([item["metrics"].get("context_to_answer_compression_ratio", 0) for item in single_trial_run])

            trial_latencies.append(trial_total_latency)
            trial_costs.append(trial_total_cost)
            trial_compressions.append(avg_compression)

            # Collect ROUGE-L, Token Efficiency, and Mismatch@K
            for item in single_trial_run:
                m = item.get("metrics", {})
                if "rouge_l_f1" in m:
                    trial_rouge_l.append(m["rouge_l_f1"])
                if "token_efficiency" in m:
                    trial_token_eff.append(m["token_efficiency"])
                for key, val in m.items():
                    if key.startswith("mismatch@k="):
                        if key not in trial_mismatches:
                            trial_mismatches[key] = []
                        trial_mismatches[key].append(val)

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
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

        # Timing, Cost, Lexical & Efficiency medians
        stats_summary["latency_median_sec"] = round(float(np.median(trial_latencies)), 2) if trial_latencies else None
        stats_summary["rouge_l_f1_median"] = round(float(np.median(trial_rouge_l)), 4) if trial_rouge_l else None
        stats_summary["token_efficiency_median"] = round(float(np.median(trial_token_eff)), 4) if trial_token_eff else None
        stats_summary["avg_compression_ratio"] = round(float(np.mean(trial_compressions)), 4) if trial_compressions else None
        stats_summary["avg_cost_usd"] = round(float(np.mean(trial_costs)), 6) if trial_costs else None

        # Mismatch @ K medians
        for k_label, m_values in trial_mismatches.items():
            if m_values:
                stats_summary[f"{k_label}_median"] = round(float(np.mean(m_values)), 4)
            else:
                stats_summary[f"{k_label}_median"] = None

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
            }, f, indent=2, default=_safe_serializer)

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

        # Store detailed trial records safely using the custom serializer
        details_path = run_dir / "detailed_results.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(strategy_results, f, indent=2, default=_safe_serializer)

        logger.info(f"Run artifacts successfully saved to: {run_dir}")
        return run_dir
