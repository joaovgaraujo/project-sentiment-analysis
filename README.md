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
Epoch   1 | train error: 0.6554 | test error: 0.7627
Epoch  10 | train error: 0.3629 | test error: 0.5628
...
Epoch 100 | train error: 0.1811 | test error: 0.3224

========================================
  Classification Metrics Report
========================================
  accuracy    : 0.9242
  f1_score    : 0.9604
  precision   : 0.9238
  recall      : 1.0000
========================================
```

The trained model is saved to `results/model.pt` and its metrics to
`results/metrics/metrics.json`.

### 4. Run tests

The test suite uses Python's built-in `unittest`, so no extra packages are needed.

```bash
python -m unittest discover -s tests
```

Expected output (training logs omitted):

```
----------------------------------------------------------------------
Ran 44 tests in 1.1s

OK
```

The suite covers data loading, preprocessing output, tensor shape, model output,
and model save/load, plus a full pipeline integration test.

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
├── notebooks/
│   └── requisitos_e_objetivos.ipynb  # Requirements and objectives (Entrega 4)
├── tests/
│   ├── test_data_loader.py        # Data loader unit tests
│   ├── test_preprocessing.py      # Preprocessing unit tests
│   ├── test_model.py              # Model shape, accuracy, save/load
│   ├── test_predict.py            # Prediction unit tests
│   ├── test_evaluation.py         # Metrics unit tests
│   ├── test_split.py              # Train/test split tests
│   ├── test_properties.py         # Property tests across many inputs
│   └── test_integration.py        # Full pipeline integration test
└── src/
    ├── data/
    │   └── loader.py              # CSV loading and schema validation
    ├── preprocessing/
    │   └── transform.py           # Text cleaning, labels, vectorization
    ├── models/
    │   └── model.py               # PyTorch model training/prediction/saving
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
train_model()          # PyTorch logistic regression (prints train/test error)
    │
    ▼
predict()              # np.ndarray of 0 (NEGATIVE) or 1 (POSITIVE)
    │
    ▼
evaluate_model()       # accuracy, F1, precision, recall (NumPy)
    │
    ▼
save_model()           # results/model.pt + results/metrics/metrics.json
```

## Current Model

| Component     | Choice                                    |
|---------------|-------------------------------------------|
| Features      | Bag-of-Words (NumPy count matrix)         |
| Classifier    | Logistic regression (PyTorch)             |
| Training      | SGD + BCEWithLogitsLoss, 100 epochs       |
| Evaluation    | Accuracy, F1, Precision, Recall (NumPy)   |
| Split         | 80/20 train/test (NumPy random shuffle)   |

## PyTorch Implementation (Entrega 3)

The classifier is a single linear layer (`nn.Linear`) trained with gradient
descent. This deliverable covers four steps:

1. **Load the data** — `load_data()` reads and validates the CSV.
2. **Train the data** — `train_model()` runs the PyTorch training loop.
3. **Print training and test error** — the loss on both sets is printed every
   10 epochs so learning can be followed.
4. **Save the results** — the model weights go to `results/model.pt` and the
   metrics to `results/metrics/metrics.json`.
