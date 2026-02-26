"""
Factuality detection model pipeline - extracted for Streamlit demo.
Matches the pipeline in Data4Good_Case_Challenge_Colab.ipynb
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def extract_features(df):
    """Extract features from question, context, and answer."""
    features = df.copy()
    
    # Basic text features
    features['question_length'] = features['question'].str.len()
    features['context_length'] = features['context'].str.len()
    features['answer_length'] = features['answer'].str.len()
    
    features['question_words'] = features['question'].str.split().str.len()
    features['context_words'] = features['context'].str.split().str.len()
    features['answer_words'] = features['answer'].str.split().str.len()
    
    # Ratio features
    features['answer_to_question_ratio'] = features['answer_length'] / (features['question_length'] + 1)
    features['answer_to_context_ratio'] = features['answer_length'] / (features['context_length'] + 1)
    features['question_to_context_ratio'] = features['question_length'] / (features['context_length'] + 1)
    
    def word_overlap(text1, text2):
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        if len(words1) == 0 or len(words2) == 0:
            return 0
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    features['question_answer_overlap'] = features.apply(
        lambda x: word_overlap(x['question'], x['answer']), axis=1
    )
    features['context_answer_overlap'] = features.apply(
        lambda x: word_overlap(x['context'], x['answer']), axis=1
    )
    features['question_context_overlap'] = features.apply(
        lambda x: word_overlap(x['question'], x['context']), axis=1
    )
    
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose']
    for qw in question_words:
        features[f'has_{qw}'] = features['question'].str.lower().str.contains(qw, regex=False).astype(int)
    
    features['answer_starts_with_question_word'] = features.apply(
        lambda x: any(str(x['answer']).lower().startswith(qw) for qw in question_words), axis=1
    ).astype(int)
    
    features['question_sentences'] = features['question'].str.count(r'[.!?]+')
    features['context_sentences'] = features['context'].str.count(r'[.!?]+')
    features['answer_sentences'] = features['answer'].str.count(r'[.!?]+')
    
    features['answer_caps_ratio'] = features['answer'].str.findall(r'[A-Z]').str.len() / (features['answer_length'] + 1)
    features['question_caps_ratio'] = features['question'].str.findall(r'[A-Z]').str.len() / (features['question_length'] + 1)
    
    features['answer_special_chars'] = features['answer'].str.findall(r'[^a-zA-Z0-9\s]').str.len()
    features['question_special_chars'] = features['question'].str.findall(r'[^a-zA-Z0-9\s]').str.len()
    
    features['answer_has_numbers'] = features['answer'].str.contains(r'\d', regex=True).astype(int)
    features['question_has_numbers'] = features['question'].str.contains(r'\d', regex=True).astype(int)
    features['context_has_numbers'] = features['context'].str.contains(r'\d', regex=True).astype(int)
    
    return features


def add_semantic_similarity(train_features, test_features, vectorizer=None):
    """Add semantic_similarity feature using TF-IDF and cosine similarity."""
    train_features = train_features.copy()
    test_features = test_features.copy()
    
    train_features['question_context_combined'] = train_features['question'] + ' ' + train_features['context']
    test_features['question_context_combined'] = test_features['question'] + ' ' + test_features['context']
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        train_qc = vectorizer.fit_transform(train_features['question_context_combined'])
    else:
        train_qc = vectorizer.transform(train_features['question_context_combined'])
    
    train_answer = vectorizer.transform(train_features['answer'])
    test_qc = vectorizer.transform(test_features['question_context_combined'])
    test_answer = vectorizer.transform(test_features['answer'])
    
    train_features['semantic_similarity'] = [
        cosine_similarity(train_qc[i:i+1], train_answer[i:i+1])[0][0]
        for i in range(len(train_features))
    ]
    test_features['semantic_similarity'] = [
        cosine_similarity(test_qc[i:i+1], test_answer[i:i+1])[0][0]
        for i in range(len(test_features))
    ]
    
    return train_features, test_features, vectorizer


def add_semantic_similarity_single(features_df, vectorizer):
    """Add semantic_similarity for a single row or few rows (for prediction)."""
    features = features_df.copy()
    features['question_context_combined'] = features['question'] + ' ' + features['context']
    qc_vec = vectorizer.transform(features['question_context_combined'])
    answer_vec = vectorizer.transform(features['answer'])
    features['semantic_similarity'] = [
        cosine_similarity(qc_vec[i:i+1], answer_vec[i:i+1])[0][0]
        for i in range(len(features))
    ]
    return features


def train_model(train_df, sample_size=5000):
    """Train the ensemble model on a sample of the data for demo."""
    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    train_features = extract_features(train_df)
    test_dummy = train_df.head(1).copy()  # For vectorizer fitting
    test_features = extract_features(test_dummy)
    
    train_features, _, vectorizer = add_semantic_similarity(train_features, test_features)
    
    feature_columns = [c for c in train_features.columns 
                      if c not in ['question', 'context', 'answer', 'type', 'question_context_combined', 'ID']]
    
    X = train_features[feature_columns].fillna(0)
    y = train_features['type']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', use_label_encoder=False)
        estimators = [('rf', rf_model), ('xgb', xgb_model), ('lr', lr_model)]
        weights = [2, 2, 1]
    else:
        estimators = [('rf', rf_model), ('lr', lr_model)]
        weights = [2, 1]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights
    )
    ensemble.fit(X_scaled, y_encoded)
    
    return {
        'model': ensemble,
        'scaler': scaler,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns
    }


def predict(pipeline, question, context, answer):
    """Make prediction for a single question-context-answer."""
    df = pd.DataFrame([{
        'question': question,
        'context': context,
        'answer': answer
    }])
    
    features = extract_features(df)
    features_with_sim = add_semantic_similarity_single(features, pipeline['vectorizer'])
    
    X = features_with_sim[pipeline['feature_columns']].fillna(0)
    X_scaled = pipeline['scaler'].transform(X)
    
    pred_encoded = pipeline['model'].predict(X_scaled)
    pred = pipeline['label_encoder'].inverse_transform(pred_encoded)
    
    proba = pipeline['model'].predict_proba(X_scaled)[0]
    labels = pipeline['label_encoder'].classes_
    
    return pred[0], dict(zip(labels, proba.tolist()))


def save_pipeline(pipeline, path="model_pipeline.joblib"):
    """Save trained pipeline to disk."""
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path="model_pipeline.joblib"):
    """Load pipeline from disk. Returns None if file not found."""
    if os.path.exists(path):
        return joblib.load(path)
    return None
