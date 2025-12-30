# NLP Sentiment Analysis on Twitter Data

This project implements an end-to-end sentiment analysis pipeline on Twitter text using both classical machine learning and deep learning approaches. A TF-IDF + Logistic Regression baseline is built first, followed by an LSTM-based model to capture contextual sentiment.

## Project Goal

- Build a reliable sentiment classification model for short text
- Compare classical NLP methods with deep learning models
- Understand trade-offs between model complexity and performance

## Dataset

- Sentiment140 (Twitter Sentiment Dataset)
- Approximately 1.6 million tweets
- Binary labels:
  - 0: Negative
  - 1: Positive

## Workflow

### 1. Data Understanding and EDA
- Dataset inspection and label mapping
- Duplicate tweet removal
- Text EDA including tweet length, word count, hashtags, mentions, and URLs
- Insights used to guide preprocessing decisions

### 2. Text Preprocessing
- Lowercasing
- URL and mention removal
- Punctuation handling
- Stopword removal with negation preservation
- Creation of a cleaned text column
- Manual validation of preprocessing output

### 3. Classical NLP Baseline
- TF-IDF vectorization using unigrams and bigrams
- Vocabulary size limited to 50,000
- Logistic Regression with class balancing
- Proper train-validation split

Baseline performance:
- Accuracy approximately 80%
- Balanced precision, recall, and F1-score

### 4. Error Analysis
- Confusion matrix inspection
- Qualitative analysis of misclassified tweets
- Identified limitations of bag-of-words models:
  - implicit sentiment
  - informal language
  - context dependency
  - negation handling

### 5. Deep Learning Model
- Tokenization and sequence padding
- Data-driven sequence length selection
- Trainable word embeddings
- Lightweight LSTM architecture
- Early stopping for regularization

LSTM performance:
- Validation accuracy approximately 82.6%
- Macro F1-score approximately 0.83

### 6. Model Comparison

| Model | Accuracy | Macro F1 |
|------|----------|----------|
| TF-IDF + Logistic Regression | ~0.80 | ~0.80 |
| LSTM | ~0.83 | ~0.83 |

## Key Takeaways

- Classical NLP methods are strong baselines for short-text sentiment analysis
- LSTM provides consistent improvement by modeling word order and context
- Deep learning should be applied when justified by error analysis
- Model selection depends on performance needs and complexity constraints

## Tools and Libraries

- Python
- NumPy
- Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

## Usage

The trained model can be used for offline sentiment analysis on large text datasets such as customer feedback, reviews, or social media posts. No API or frontend is required.

## Author

Astitva Mishra
