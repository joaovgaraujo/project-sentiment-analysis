# Sentiment Analysis — Product Reviews

An Artificial Intelligence application for **binary sentiment classification** of product reviews. Given a review text, the system predicts whether the sentiment is **positive** or **negative**.

## Problem

Customer reviews on e-commerce platforms contain valuable information, but their volume makes manual reading unfeasible. This project builds an NLP pipeline to automatically classify the sentiment of textual reviews, handling noise, informal language, and linguistic variations.

Classification is **binary**:
- **Positive (1)**: rating ≥ 4 stars
- **Negative (0)**: rating ≤ 2 stars
- Rating 3 (neutral/ambiguous) is discarded

## Dataset

**Amazon Product Reviews** — [Kaggle: yasserh/amazon-product-reviews-dataset](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset)

License: CC0-1.0

Columns used:
- `reviews.text`: review text
- `reviews.rating`: rating from 1 to 5

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/anapaulabarros/project-sentiment-analysis.git
cd project-sentiment-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the dataset

**Option A — Kaggle CLI** (requires `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d yasserh/amazon-product-reviews-dataset -p data/raw/
unzip data/raw/amazon-product-reviews-dataset.zip -d data/raw/
mv data/raw/7817_1.csv data/raw/reviews.csv
rm data/raw/amazon-product-reviews-dataset.zip
```

**Windows (PowerShell):**

```powershell
pip install kaggle
kaggle datasets download -d yasserh/amazon-product-reviews-dataset -p data/raw/
Expand-Archive -Path "data/raw/amazon-product-reviews-dataset.zip" -DestinationPath "data/raw/" -Force
Rename-Item "data/raw/7817_1.csv" "reviews.csv"
Remove-Item "data/raw/amazon-product-reviews-dataset.zip"
```

**Option B — Manual download:**

1. Go to https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset
2. Click "Download" and extract the ZIP
3. Rename `7817_1.csv` to `reviews.csv`
4. Place it at `data/raw/reviews.csv`

### 3. Run the pipeline

```bash
python main.py
```

Expected output:
```
========================================
  Classification Metrics Report
========================================
  accuracy    : 0.9190
  f1_score    : 0.9570
  precision   : 0.9356
  recall      : 0.9793
========================================
```

### 4. Run tests

```bash
pip install pytest
python -m pytest
```

All 115 tests should pass.

## Project Structure

```
project-sentiment-analysis/
├── main.py                        # Central pipeline entry point
├── requirements.txt
├── data/
│   ├── raw/                       # Original dataset (not versioned)
│   └── processed/                 # Preprocessed data
├── results/
│   ├── metrics/                   # Evaluation metrics per experiment
│   └── figures/                   # Plots and visualizations
├── tests/
│   ├── test_data_loader.py        # Data loader unit tests
│   ├── test_preprocessing.py      # Preprocessing unit tests
│   ├── test_predict.py            # Prediction unit tests
│   ├── test_evaluation.py         # Metrics unit tests
│   ├── test_split.py              # Train/test split tests
│   ├── test_properties.py         # Property-based tests
│   └── test_integration.py        # Full pipeline integration test
└── src/
    ├── data/
    │   └── loader.py              # CSV loading and schema validation
    ├── preprocessing/
    │   └── transform.py           # Text cleaning, labels, vectorization
    ├── models/
    │   └── model.py               # LogisticRegression training/prediction
    ├── training/
    │   └── train.py               # Dataset splitting and orchestration
    ├── evaluation/
    │   └── metrics.py             # NumPy-based metrics and reporting
    └── utils/
        └── config.py              # Global constants and parameters
```

## NLP Pipeline

```
CSV file (data/raw/reviews.csv)
    │
    ▼
load_data()            # load CSV, validate columns
    │
    ▼
preprocess_dataset()   # clean_text + normalize_label + drop neutrals
    │
    ▼
split_dataset()        # NumPy-based 80/20 train/test split
    │
    ▼
build_vocabulary()     # word → index mapping from training texts
    │
    ▼
texts_to_matrix()      # NumPy Bag-of-Words count matrix
    │
    ▼
train_model()          # LogisticRegression on NumPy arrays
    │
    ▼
predict()              # np.ndarray of 0 (NEGATIVE) or 1 (POSITIVE)
    │
    ▼
evaluate_model()       # accuracy, F1, precision, recall (NumPy)
```

## Current Model

| Component     | Choice                                    |
|---------------|-------------------------------------------|
| Features      | Bag-of-Words (NumPy count matrix)         |
| Classifier    | LogisticRegression (scikit-learn)          |
| Evaluation    | Accuracy, F1, Precision, Recall (NumPy)   |
| Split         | 80/20 train/test (NumPy random shuffle)   |

## Course Deliverables Covered (Stage 1)

1. **Functions & modularization** — all code in small, focused functions
2. **Package structure** — `src/data`, `src/preprocessing`, `src/models`, `src/training`, `src/evaluation`, `src/utils`
3. **Type hints** — all functions annotated with parameter and return types
4. **NumPy** — vocabulary building, count matrix vectorization, train/test split, evaluation metrics

Deep learning models with PyTorch will be introduced in Stage 2 (deliverables 5–6).
