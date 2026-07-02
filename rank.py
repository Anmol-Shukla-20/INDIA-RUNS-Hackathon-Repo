import os
import time
import pickle
import pandas as pd
from scoring import calculate_textual_similarity, compute_heuristic_score
from reasoning import LocalReasoningGenerator

JD_TEXT = """
Deep technical depth in modern ML systems embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude willing to ship a working ranker in a week.
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5)
Production experience with vector databases or hybrid search infrastructure Pinecone, Weaviate, Qdrant, Milvus, FAISS
Strong Python code quality.
Evaluation frameworks for ranking systems NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.
"""

def main():
    start_time = time.time()
    print("Starting Ranking System...")
    
    # Load Artifacts
    if not os.path.exists('artifacts/vectorizer.pkl'):
        print("Error: Artifacts not found. Please run precompute.py first.")
        return
        
    print("Loading precomputed features...")
    with open('artifacts/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open('artifacts/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    df = pd.read_parquet('artifacts/candidates_features.parquet')
    
    # 1. Calculate base textual similarity
    print("Computing textual match to JD...")
    text_sims = calculate_textual_similarity(JD_TEXT, vectorizer, tfidf_matrix)
    
    # 2. Compute composite heuristic score
    print("Applying heuristic scoring...")
    final_scores = compute_heuristic_score(df, text_sims)
    df['score'] = final_scores
    
    # 3. Sort and get Top 100
    print("Sorting candidates...")
    top_100 = df.sort_values(by='score', ascending=False).head(100).copy()
    top_100['rank'] = range(1, 101)
    
    # 4. Generate Reasoning
    print("Initializing local LLM for reasoning generation...")
    # Initialize the LLM (downloads weights if not cached)
    llm = LocalReasoningGenerator()
    
    reasonings = []
    print("Generating reasonings for Top 100...")
    for idx, row in top_100.iterrows():
        reasoning = llm.generate_reasoning(
            candidate={'profile': {'years_of_experience': row['years_of_experience']}, 
                       'redrob_signals': {'notice_period_days': row['notice_period_days']}}, 
            jd_summary=JD_TEXT, 
            rank=row['rank']
        )
        reasonings.append(reasoning)
        
    top_100['reasoning'] = reasonings
    
    # 5. Format Submission
    submission = top_100[['candidate_id', 'rank', 'score', 'reasoning']]
    submission.to_csv('submission.csv', index=False, encoding='utf-8')
    
    elapsed = time.time() - start_time
    print(f"Ranking complete! Output saved to submission.csv. Time elapsed: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
