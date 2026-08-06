import pandas as pd
from typing import List, Dict, Any
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
        # ROUGE-L evaluates how well generated summaries follow structural flow
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
        ground_truth: str,
        latency: float,
        token_efficiency: float
    ) -> Dict[str, float]:
        """Runs the LLM-as-a-judge and deterministic metrics simultaneously."""
        
        # 1. Prepare Ragas Dataset Schema
        # Note: Depending on specific Ragas v0.1.x, 'ground_truth' vs 'ground_truths' is required. 
        # v0.1.7 uses a single string list for 'ground_truth'.
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        # 2. Run LLM-as-a-judge metrics
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            # Silences the progress bar for cleaner CLI output
            raise_exceptions=False 
        )
        
        # 3. Compute Deterministic Metrics
        rouge_l_score = self.evaluate_rouge_l(answer, ground_truth)
        
        # 4. Compile the 7-metric dictionary
        return {
            "Faithfulness": float(ragas_result.get("faithfulness", 0.0)),
            "Answer Relevancy": float(ragas_result.get("answer_relevancy", 0.0)),
            "Context Precision": float(ragas_result.get("context_precision", 0.0)),
            "Context Recall": float(ragas_result.get("context_recall", 0.0)),
            "ROUGE-L": rouge_l_score,
            "Latency (s)": latency,
            "Token Efficiency": token_efficiency
        }
