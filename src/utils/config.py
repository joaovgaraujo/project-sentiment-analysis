"""Global constants and configuration parameters for the project."""

# Dataset columns (Amazon Product Reviews - Kaggle: yasserh)
TEXT_COLUMN: str = "reviews.text"
RATING_COLUMN: str = "reviews.rating"
LABEL_COLUMN: str = "sentiment"

# Sentiment derivation from rating
MIN_POSITIVE_RATING: int = 4  # rating >= 4 → positive (1)
MAX_NEGATIVE_RATING: int = 2  # rating <= 2 → negative (0)
# rating == 3 → neutral, discarded

# Train/test split
TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42

# PyTorch training
EPOCHS: int = 100
LEARNING_RATE: float = 0.1
BATCH_SIZE: int = 32

# Output paths for the trained model, vocabulary and metrics
MODEL_PATH: str = "results/model.pt"
VOCAB_PATH: str = "results/vocab.json"
METRICS_PATH: str = "results/metrics/metrics.json"
