"""
utils/mitigator.py
Implements two bias mitigation strategies:

1. Reweighting  — assign higher sample weights to under-represented
                  (group, label) combinations during training.
2. Remove Sensitive Feature — train without the sensitive column at all.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def apply_reweighting(
    X_train:        np.ndarray,
    y_train:        np.ndarray,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    sensitive_test: np.ndarray,
    feature_names:  list,
    sensitive_attr: str = "gender",
):
    """
    Pre-processing: reweight training samples so that every
    (group × label) cell is equally represented.

    Returns (new_model, X_test, y_test)  — test set unchanged.
    """
    sens_col_idx = (feature_names.index(sensitive_attr)
                    if sensitive_attr in feature_names else 0)
    sensitive_train = X_train[:, sens_col_idx]

    groups = np.unique(sensitive_train)
    weights = np.ones(len(y_train))

    for g in groups:
        for label in [0, 1]:
            mask = (sensitive_train == g) & (y_train == label)
            n_cell = mask.sum()
            if n_cell == 0:
                continue
            # Weight = (expected uniform share) / (actual share)
            expected = len(y_train) / (len(groups) * 2)
            weights[mask] = expected / n_cell

    new_model = LogisticRegression(max_iter=1000, solver="lbfgs",
                                   random_state=42, class_weight="balanced")
    new_model.fit(X_train, y_train, sample_weight=weights)
    return new_model, X_test, y_test


def apply_remove_sensitive(
    X_train:        np.ndarray,
    y_train:        np.ndarray,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    feature_names:  list,
    sensitive_attr: str = "gender",
):
    """
    Pre-processing: drop the sensitive feature entirely before training.

    Returns (new_model, X_test_reduced, y_test)
    """
    if sensitive_attr not in feature_names:
        # Nothing to drop — fall back to normal training
        new_model = LogisticRegression(max_iter=1000, solver="lbfgs",
                                       random_state=42, class_weight="balanced")
        new_model.fit(X_train, y_train)
        return new_model, X_test, y_test

    drop_idx = feature_names.index(sensitive_attr)
    keep_idx = [i for i in range(X_train.shape[1]) if i != drop_idx]

    X_train_r = X_train[:, keep_idx]
    X_test_r  = X_test[:, keep_idx]

    new_model = LogisticRegression(max_iter=1000, solver="lbfgs",
                                   random_state=42, class_weight="balanced")
    new_model.fit(X_train_r, y_train)
    return new_model, X_test_r, y_test
