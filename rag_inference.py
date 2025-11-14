"""
Retrieval-Augmented Generation (RAG) for Recipe Recommendations

Author: Ata Jodeiri Seyedian
University of Oulu, 2025
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import torch


class RecipeRAG:
    """
    RAG system using FAISS semantic search to ground LLM outputs.
    """
    
    def __init__(self, recipe_df, embedder_model="multi-qa-mpnet-base-dot-v1"):
        """
        Args:
            recipe_df: DataFrame with recipe data
            embedder_model: Sentence-transformer model name
        """
        self.df = recipe_df
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load embedder
        self.embedder = SentenceTransformer(embedder_model, device=self.device)
        
        # Format recipes for search
        self.recipe_texts = self._format_recipes()
        
        # Build search index
        self.index = None
        self.embeddings = None
    
    def _format_recipes(self):
        """Format recipes into searchable text."""
        texts = []
        for _, row in self.df.iterrows():
            text = (
                f"Title: {row.get('title', '')}\n"
                f"Tags: {row.get('tags', '')}\n"
                f"Nutrition: {row.get('calories [cal]', '')} kcal, "
                f"{row.get('protein [g]', '')}g protein, "
                f"{row.get('sodium [mg]', '')}mg sodium"
            )
            texts.append(text)
        return texts
    
    def build_index(self):
        """Build FAISS index from recipe embeddings."""
        print("Encoding recipes for FAISS index...")
        
        # Generate embeddings
        self.embeddings = self.embedder.encode(
            self.recipe_texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Create FAISS index (cosine similarity)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"✅ Built index with {self.index.ntotal} recipes")
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k most relevant recipes.
        
        Args:
            query: User instruction/query
            top_k: Number of recipes to retrieve
            
        Returns:
            List of retrieved recipe texts
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return retrieved texts
        retrieved = []
        for idx in indices[0]:
            if idx < len(self.recipe_texts):
                retrieved.append(self.recipe_texts[idx])
        
        return retrieved
    
    def create_grounded_prompt(self, instruction, retrieved_recipes):
        """
        Create prompt with retrieved context for grounding.
        
        Args:
            instruction: User instruction
            retrieved_recipes: List of retrieved recipe texts
            
        Returns:
            Grounded prompt string
        """
        context = "\n\n".join([f"Recipe {i+1}:\n{r}" 
                               for i, r in enumerate(retrieved_recipes)])
        
        prompt = f"""### Context:
{context}

### Instruction:
{instruction}

### Constraint:
Only use information from the context above.
List real recipes and include accurate nutrition values.

### Assistant:
"""
        return prompt


# Example usage
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("pp_recipes.csv", low_memory=False)
    df = df.dropna(subset=["title"]).head(1000)  # Sample for demo
    
    # Initialize RAG
    rag = RecipeRAG(df)
    rag.build_index()
    
    # Example query
    query = "Suggest a high-protein recipe under 400 calories"
    
    # Retrieve relevant recipes
    retrieved = rag.retrieve(query, top_k=3)
    
    # Create grounded prompt
    prompt = rag.create_grounded_prompt(query, retrieved)
    
    print("Retrieved recipes:")
    for i, recipe in enumerate(retrieved, 1):
        print(f"\n{i}. {recipe[:100]}...")
    
    # Use prompt with your LLM for generation
    # response = model.generate(prompt)
