"""
utils/model_trainer.py
Trains a Logistic Regression model — chosen for interpretability.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Fit a Logistic Regression model.

    Why LR?
    - Coefficients are directly interpretable
    - Works well with SHAP LinearExplainer
    - Fast to train — ideal for hackathon demos
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",   # handles class imbalance
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> float:
    """Return overall accuracy on the test set."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
