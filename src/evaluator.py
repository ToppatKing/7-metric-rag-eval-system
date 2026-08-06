import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate_rouge_l(self, generated_answer: str, ground_truth: str) -> float:
        """Computes structural similarity using Longest Common Subsequence F1."""
        scores = self.rouge_scorer.score(ground_truth, generated_answer)
        return scores['rougeL'].fmeasure

    def run_evaluation(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        latency: float,
        token_efficiency: float,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs evaluation. Skips Recall and ROUGE-L if ground_truth is absent."""
        
        # 1. Prepare Base Ragas Dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        
        # 2. Dynamically assign metrics
        metrics_to_run = [faithfulness, answer_relevancy, context_precision]
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics_to_run.append(context_recall)

        dataset = Dataset.from_dict(data)

        # 3. Run LLM-as-a-judge metrics
        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics_to_run,
            raise_exceptions=False 
        )
        
        # 4. Handle Conditional Metrics
        if ground_truth:
            rouge_l_score = self.evaluate_rouge_l(answer, ground_truth)
            c_recall = float(ragas_result.get("context_recall", 0.0))
        else:
            rouge_l_score = "N/A"
            c_recall = "N/A"
        
        # 5. Compile the final metric dictionary
        return {
            "Faithfulness": float(ragas_result.get("faithfulness", 0.0)),
            "Answer Relevancy": float(ragas_result.get("answer_relevancy", 0.0)),
            "Context Precision": float(ragas_result.get("context_precision", 0.0)),
            "Context Recall": c_recall,
            "ROUGE-L": rouge_l_score,
            "Latency (s)": latency,
            "Token Efficiency": token_efficiency
        }
