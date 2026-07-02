# Redrob AI Hackathon Submission: Intelligent Candidate Ranking

This repository contains our submission for the Intelligent Candidate Discovery & Ranking Challenge.

## Architecture

Our approach is designed to be extremely fast and efficient, fitting within the 5-minute CPU constraint while maintaining high precision.
We use a **two-step architecture**:

1. **Pre-computation (Offline)**: We process `candidates.jsonl`, filter out honeypot profiles, extract text features (experience, skills, headlines), and build a TF-IDF index. We also extract behavioral signals.
2. **Ranking & Reasoning (Online)**: We query the precomputed index, apply a weighted heuristic score that rewards JD-specific text matches and behavioral engagement, and use a localized lightweight LLM (`Qwen1.5-0.5B-Chat`) to generate personalized reasoning for the top 100.

## Reproducing the Submission

To bypass local hardware constraints, we recommend running this pipeline on a **Google Colab** CPU instance. It perfectly emulates the 16GB sandbox environment.

### Step 1: Setup
1. Clone this repository or copy the code to a Google Colab notebook.
2. Ensure `candidates.jsonl` is in the root directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Pre-compute (Run Once)
This step does not count towards the 5-minute ranking constraint. It processes the dataset and creates artifacts.
```bash
python src/precompute.py
```
*This will create an `artifacts/` folder containing the vectorizer and feature data.*

### Step 3: Fast Ranking (< 5 Minutes)
This is the main step that produces the submission CSV.
```bash
python src/rank.py
```
*This will output `submission.csv`.*

## Evaluating
You can run the official validator on the output:
```bash
python validate_submission.py submission.csv
```
