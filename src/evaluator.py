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
        ground_truth: str = None,
        latency: float = 0.0,
        token_efficiency: float = 0.0
    ) -> Dict[str, float]:
        
        has_gt = bool(ground_truth and ground_truth.strip())
        
        # Select metrics dynamically based on ground truth availability
        metrics_to_eval = [faithfulness, answer_relevancy]
        if has_gt:
            metrics_to_eval.extend([context_precision, context_recall])
            
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        if has_gt:
            data["ground_truth"] = [ground_truth]
            
        dataset = Dataset.from_dict(data)
        
        ragas_result = evaluate(dataset=dataset, metrics=metrics_to_eval, raise_exceptions=False)
        
        results = {
            "Faithfulness": float(ragas_result.get("faithfulness", 0.0)),
            "Answer Relevancy": float(ragas_result.get("answer_relevancy", 0.0)),
            "Latency (s)": latency,
            "Token Efficiency": token_efficiency
        }
        
        # Conditionally compute ground-truth dependent metrics
        if has_gt:
            results["Context Precision"] = float(ragas_result.get("context_precision", 0.0))
            results["Context Recall"] = float(ragas_result.get("context_recall", 0.0))
            results["ROUGE-L"] = self.evaluate_rouge_l(answer, ground_truth)
        else:
            results["Context Precision"] = float('nan')
            results["Context Recall"] = float('nan')
            results["ROUGE-L"] = float('nan')
            
        return results
