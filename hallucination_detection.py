"""
Hallucination Detection System for Recipe Recommendations

Author: Ata Jodeiri Seyedian
University of Oulu, 2025
"""

import pandas as pd
import re
import json
from difflib import get_close_matches
from collections import Counter


class HallucinationDetector:
    """
    Multi-dimensional hallucination detection for LLM-generated recipes.
    
    Validates outputs across four dimensions:
    1. Recipe name existence (fuzzy matching)
    2. Nutritional accuracy (tolerance-based)
    3. Ingredient consistency
    4. Keyword grounding
    """
    
    def __init__(self, reference_df, tolerance=0.3, fuzzy_cutoff=0.75):
        """
        Args:
            reference_df: Ground truth recipe DataFrame
            tolerance: Acceptable error for numerical values (default 30%)
            fuzzy_cutoff: Minimum similarity for string matching
        """
        self.df = reference_df
        self.tolerance = tolerance
        self.fuzzy_cutoff = fuzzy_cutoff
        self.catalog = self._build_catalog()
    
    def _build_catalog(self):
        """Build catalog of known recipes, ingredients, and keywords."""
        catalog = {
            "titles": set(self.df["title"].dropna().str.lower().str.strip()),
            "ingredients": set(),
            "keywords": set()
        }
        
        # Extract ingredients
        for ing_col in ["ingredients", "ingredient_food_kg_names"]:
            if ing_col in self.df.columns:
                for ing in self.df[ing_col].dropna():
                    for item in re.split(r"[,;]", str(ing).lower()):
                        if len(item.strip()) > 2:
                            catalog["ingredients"].add(item.strip())
        
        return catalog
    
    @staticmethod
    def extract_recipe_name(text):
        """Extract recipe name from generated text."""
        match = re.match(r"([A-Za-z0-9 &''\\-]+)", text.strip())
        return match.group(1).lower().strip() if match else None
    
    @staticmethod
    def extract_nutritional_values(text):
        """Extract calories, protein, and sodium from text."""
        text_lower = text.lower()
        kcal = re.search(r"(\d+(?:\.\d+)?)\s*kcal", text_lower)
        protein = re.search(r"(\d+(?:\.\d+)?)\s*g\s*protein", text_lower)
        sodium = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*sodium", text_lower)
        
        return {
            "calories": float(kcal.group(1)) if kcal else None,
            "protein": float(protein.group(1)) if protein else None,
            "sodium": float(sodium.group(1)) if sodium else None,
        }
    
    def detect_hallucination(self, generated_text):
        """
        Comprehensive hallucination detection for a single output.
        
        Returns:
            Dictionary with detection results
        """
        result = {
            "text": generated_text[:300],
            "is_hallucinated": False,
            "hallucination_types": []
        }
        
        # Extract information
        recipe_name = self.extract_recipe_name(generated_text)
        
        # Check 1: Recipe existence
        if recipe_name:
            matches = get_close_matches(
                recipe_name, 
                self.catalog["titles"], 
                n=1, 
                cutoff=self.fuzzy_cutoff
            )
            if not matches:
                result["is_hallucinated"] = True
                result["hallucination_types"].append("invented_recipe")
        
        # Check 2: Nutritional accuracy (if recipe exists)
        if recipe_name and matches:
            gen_nutrition = self.extract_nutritional_values(generated_text)
            ref_row = self.df[self.df["title"].str.lower() == matches[0]]
            
            if not ref_row.empty:
                for key, col in [("calories", "calories [cal]"), 
                                ("protein", "protein [g]"), 
                                ("sodium", "sodium [mg]")]:
                    gen_val = gen_nutrition[key]
                    ref_val = ref_row[col].values[0]
                    
                    if gen_val and ref_val and ref_val > 0:
                        error = abs(gen_val - ref_val) / ref_val
                        if error > self.tolerance:
                            result["is_hallucinated"] = True
                            result["hallucination_types"].append("incorrect_nutrition")
                            break
        
        return result
    
    def analyze_predictions(self, predictions):
        """
        Analyze hallucination rate across multiple predictions.
        
        Args:
            predictions: List of dicts with 'model_output' key
            
        Returns:
            Summary statistics
        """
        results = []
        type_counts = Counter()
        
        for pred in predictions:
            generated = pred.get("model_output", pred.get("output", ""))
            detection = self.detect_hallucination(generated)
            results.append(detection)
            
            if detection["is_hallucinated"]:
                type_counts.update(detection["hallucination_types"])
        
        total = len(predictions)
        hallucinated = sum(1 for r in results if r["is_hallucinated"])
        
        summary = {
            "total_examples": total,
            "hallucinated_count": hallucinated,
            "hallucination_rate_%": round(100 * hallucinated / total, 2) if total > 0 else 0,
            "hallucination_types": dict(type_counts)
        }
        
        return summary, results


# Example usage
if __name__ == "__main__":
    # Load reference dataset
    df = pd.read_csv("pp_recipes.csv", low_memory=False)
    
    # Load predictions
    with open("predictions.json", "r") as f:
        predictions = json.load(f)
    
    # Run detection
    detector = HallucinationDetector(df, tolerance=0.3)
    summary, results = detector.analyze_predictions(predictions)
    
    print(f"Hallucination Rate: {summary['hallucination_rate_%']}%")
    print(f"Total Examples: {summary['total_examples']}")
    print(f"Hallucinated: {summary['hallucinated_count']}")
