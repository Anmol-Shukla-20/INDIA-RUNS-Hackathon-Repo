import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from tqdm import tqdm

def is_honeypot(candidate):
    # Rule 1: Expert proficiency with 0 months duration for multiple skills
    skills = candidate.get('skills', [])
    expert_zero_months = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
    if expert_zero_months >= 3:
        return True
    
    # Rule 2: Too many years of experience vs timeline. 
    # (Simplified: just checking if any single job duration is absurdly high > 40 years)
    for job in candidate.get('career_history', []):
        if job.get('duration_months', 0) > 480: # 40 years at one company is extremely rare for tech
            return True
            
    return False

def extract_features(candidate):
    # Combine text for TF-IDF
    profile = candidate.get('profile', {})
    text_parts = [
        profile.get('headline', ''),
        profile.get('summary', ''),
        profile.get('current_title', '')
    ]
    
    for job in candidate.get('career_history', []):
        text_parts.append(job.get('title', ''))
        text_parts.append(job.get('description', ''))
        
    for skill in candidate.get('skills', []):
        text_parts.append(skill.get('name', ''))
        
    combined_text = " ".join(filter(None, text_parts)).lower()
    
    # Extract behavioral signals
    signals = candidate.get('redrob_signals', {})
    
    return {
        'candidate_id': candidate['candidate_id'],
        'combined_text': combined_text,
        'years_of_experience': profile.get('years_of_experience', 0),
        'recruiter_response_rate': signals.get('recruiter_response_rate', 0.0),
        'github_activity_score': signals.get('github_activity_score', -1.0),
        'last_active_date': signals.get('last_active_date', '2000-01-01'),
        'open_to_work_flag': signals.get('open_to_work_flag', False),
        'interview_completion_rate': signals.get('interview_completion_rate', 0.0),
        'notice_period_days': signals.get('notice_period_days', 90)
    }

def main():
    print("Starting precomputation...")
    input_file = '../candidates.jsonl'
    
    # Check if we are running in the correct directory, if not adjust path
    if not os.path.exists(input_file):
        input_file = 'candidates.jsonl'
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found. Please ensure it's in the correct directory.")
            return

    features_list = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing candidates"):
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                if is_honeypot(candidate):
                    continue # Skip honeypots entirely
                
                features = extract_features(candidate)
                features_list.append(features)
            except Exception as e:
                pass # Skip broken rows

    df = pd.DataFrame(features_list)
    print(f"Processed {len(df)} valid candidates.")
    
    print("Computing TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    print("Saving artifacts...")
    # We will save the vectorizer, the TF-IDF matrix, and the DataFrame (without the large text col)
    df_slim = df.drop(columns=['combined_text'])
    
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open('artifacts/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    df_slim.to_parquet('artifacts/candidates_features.parquet')
    
    print("Precomputation finished successfully. Artifacts saved to 'artifacts/' folder.")

if __name__ == "__main__":
    main()
