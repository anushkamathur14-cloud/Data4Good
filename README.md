# Factuality Detection in AI-Generated Educational Content

## Data4Good Competition - 4th Annual

This project implements a comprehensive machine learning solution to detect factuality in AI-generated educational content. The solution classifies answers as **Factual**, **Contradiction**, or **Irrelevant** based on questions, context, and answers.

## Project Structure

```
.
├── data/
│   ├── train.json          # Training data (21,021 examples)
│   └── test.json           # Test data (2,000 examples) - predictions will be saved here
├── Data4Good_Case_Challenge_Colab.ipynb  # Colab notebook (competition submission)
├── eda_and_ml_pipeline.ipynb  # Main Jupyter notebook with EDA and ML pipeline
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data** (if needed):
   The notebook will automatically download required NLTK data, but you can also run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Place Data Files**:
   - Ensure `data/train.json` contains the training data with columns: `Question`, `Context`, `Answer`, `Type`
   - Ensure `data/test.json` contains the test data with columns: `ID`, `Question`, `Context`, `Answer`

## Data Format

### Training Data (`data/train.json`)
```json
[
  {
    "Question": "What is the capital of France?",
    "Context": "France is a country in Europe...",
    "Answer": "Paris is the capital of France.",
    "Type": "Factual"
  },
  ...
]
```

### Test Data (`data/test.json`)
```json
[
  {
    "ID": 1,
    "Question": "What is the capital of France?",
    "Context": "France is a country in Europe...",
    "Answer": "Paris is the capital of France."
  },
  ...
]
```

## Running the Notebook

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook eda_and_ml_pipeline.ipynb
   ```

2. **Run All Cells**:
   - The notebook will automatically:
     - Load and explore the data
     - Perform EDA with visualizations
     - Engineer features
     - Train multiple ML models
     - Compare model performance
     - Generate predictions for test data
     - Save predictions to `data/test.json`

## Methodology

### Approach

1. **Exploratory Data Analysis (EDA)**:
   - Target distribution analysis
   - Text length and word count statistics
   - Visualizations by class

2. **Feature Engineering**:
   - Basic text features (length, word count)
   - Ratio features (answer/question, answer/context)
   - Word overlap features (Jaccard similarity)
   - Semantic similarity (TF-IDF + cosine similarity)
   - Question word features
   - Linguistic features (sentence count, capitalization, special characters)

3. **Machine Learning Models**:
   - **Random Forest**: Tree-based ensemble with feature importance
   - **XGBoost**: Gradient boosting for better performance
   - **Logistic Regression**: Linear baseline model
   - **Ensemble**: Voting classifier combining all models
   - **Transformer Models**: BERT/RoBERTa (optional, requires fine-tuning)

4. **Model Selection**:
   - Models evaluated on validation set
   - Best model selected based on F1 (macro) score
   - Final model retrained on full training data

5. **Predictions**:
   - Test predictions saved to `data/test.json` with `Type` column added

## Key Features

- **Comprehensive EDA**: Visualizations and statistical analysis
- **Rich Feature Engineering**: 40+ features capturing various aspects of text
- **Multiple ML Approaches**: Traditional ML and transformer-based models
- **Ensemble Methods**: Combining multiple models for robustness
- **Class Balancing**: Handling imbalanced classes
- **Detailed Methodology**: Discussion of what worked and what didn't

## Output

After running the notebook, `data/test.json` will be updated with predictions in the `Type` column:

```json
[
  {
    "ID": 1,
    "Question": "...",
    "Context": "...",
    "Answer": "...",
    "Type": "Factual"  // or "Contradiction" or "Irrelevant"
  },
  ...
]
```

## Notes

- The notebook includes transformer model setup, but fine-tuning is commented out due to computational requirements
- For production use, consider fine-tuning transformer models on the full dataset
- The ensemble model typically performs best and is used for final predictions
- All models use class balancing to handle imbalanced data

## Requirements

See `requirements.txt` for full list. Key dependencies:
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- nltk
- transformers, torch (for transformer models)

## Contact

For questions or issues, please refer to the methodology discussion section in the notebook.

