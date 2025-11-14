"""
Model Evaluation Framework for Recipe Recommendations

Author: Ata Jodeiri Seyedian
University of Oulu, 2025
"""

import json
import pandas as pd
from evaluate import load as load_metric


class ModelEvaluator:
    """
    Unified evaluation framework comparing multiple models.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.rouge = load_metric("rouge")
        self.bleu = load_metric("bleu")
    
    def evaluate_model(self, predictions, references):
        """
        Evaluate a single model's predictions.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with evaluation scores
        """
        # Text quality metrics
        rouge_scores = self.rouge.compute(predictions=predictions, references=references)
        bleu_score = self.bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        
        return {
            "rouge-1": round(rouge_scores["rouge1"], 3),
            "rouge-2": round(rouge_scores["rouge2"], 3),
            "rouge-l": round(rouge_scores["rougeL"], 3),
            "bleu": round(bleu_score["bleu"], 3)
        }
    
    def compare_models(self, model_results):
        """
        Compare multiple models side-by-side.
        
        Args:
            model_results: Dict mapping model_name -> (predictions, references)
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, (preds, refs) in model_results.items():
            scores = self.evaluate_model(preds, refs)
            scores["Model"] = model_name
            comparison_data.append(scores)
        
        df = pd.DataFrame(comparison_data)
        df = df[["Model", "rouge-l", "bleu"]]  # Reorder columns
        
        return df


# Example usage
if __name__ == "__main__":
    # Load predictions from different models
    with open("baseline_predictions.json", "r") as f:
        baseline = json.load(f)
    
    with open("finetuned_predictions.json", "r") as f:
        finetuned = json.load(f)
    
    with open("rag_predictions.json", "r") as f:
        rag = json.load(f)
    
    # Extract predictions and references
    baseline_preds = [p["model_output"] for p in baseline]
    baseline_refs = [p["reference_output"] for p in baseline]
    
    finetuned_preds = [p["model_output"] for p in finetuned]
    finetuned_refs = [p["reference_output"] for p in finetuned]
    
    rag_preds = [p["model_output"] for p in rag]
    rag_refs = [p["reference_output"] for p in rag]
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    results = evaluator.compare_models({
        "Baseline": (baseline_preds, baseline_refs),
        "Fine-Tuned": (finetuned_preds, finetuned_refs),
        "RAG + Fine-Tuned": (rag_preds, rag_refs)
    })
    
    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))
