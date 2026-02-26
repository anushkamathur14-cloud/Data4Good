"""
Pre-train the ensemble model and save to disk.
Run locally once, commit model_pipeline.joblib, then the app loads instantly.
"""
import json
import pandas as pd
from model_pipeline import train_model, save_pipeline

if __name__ == "__main__":
    print("Loading training data...")
    with open("data/train.json", "r") as f:
        train_data = json.load(f)
    train_df = pd.DataFrame(train_data)
    train_df.columns = [c.lower() for c in train_df.columns]

    print("Training model (sample_size=3000)...")
    pipeline = train_model(train_df, sample_size=3000)

    print("Saving to model_pipeline.joblib...")
    save_pipeline(pipeline)
    print("Done. Commit model_pipeline.joblib and push to GitHub.")
