import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_textual_similarity(jd_text, vectorizer, tfidf_matrix):
    """
    Computes cosine similarity between the Job Description and all candidates.
    """
    jd_vector = vectorizer.transform([jd_text.lower()])
    similarities = cosine_similarity(jd_vector, tfidf_matrix).flatten()
    return similarities

def compute_heuristic_score(df, text_similarities):
    """
    Computes the final ranking score using a weighted heuristic.
    """
    scores = np.zeros(len(df))
    
    # 1. Base score from text similarity (0 to 1) -> scaled to 100
    scores += text_similarities * 100
    
    # 2. Add points for ideal years of experience (5-9 years as per JD)
    for i, exp in enumerate(df['years_of_experience']):
        if 5 <= exp <= 9:
            scores[i] += 15
        elif 4 <= exp < 5 or 9 < exp <= 12:
            scores[i] += 5
        else:
            scores[i] -= 10
            
    # 3. Behavioral signals (Redrob signals)
    # Recruiter response rate (0 to 1)
    scores += df['recruiter_response_rate'] * 10 
    
    # Github activity score (-1 to 100) -> scale to 15 points
    github_scores = df['github_activity_score'].copy()
    github_scores[github_scores < 0] = 0 # Ignore negative (no github)
    scores += (github_scores / 100.0) * 15
    
    # Penalty for not being active recently
    # Assuming last_active_date is YYYY-MM-DD. 
    # For a hackathon, we might just penalize low response rates and low interview completion
    scores += df['interview_completion_rate'] * 5
    
    # Penalty for long notice period (>30 days starts degrading)
    notice_periods = df['notice_period_days'].values
    notice_penalty = np.where(notice_periods > 30, (notice_periods - 30) * 0.1, 0)
    scores -= notice_penalty
    
    return scores
