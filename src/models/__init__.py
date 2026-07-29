"""Model definition package."""

from src.models.model import load_model, predict, save_model, train_model

__all__ = ["train_model", "predict", "save_model", "load_model"]
