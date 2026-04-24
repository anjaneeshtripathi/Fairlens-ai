"""
utils/whatif_simulator.py
Predicts outcome for a user-defined input and then flips the sensitive
attribute (gender) to show how the model's decision changes.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


# Default values for features not set by the user
_DEFAULTS = {
    "age": 35,
    "workclass": "Private",
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "gender": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

_GENDER_FLIP = {"Male": "Female", "Female": "Male",
                "0": "1", "1": "0",
                 0: 1,  1: 0}


def predict_whatif(
    model:         LogisticRegression,
    user_input:    dict,
    feature_names: list,
    encoders:      dict,
) -> dict:
    """
    Build a feature vector from user_input, predict, then flip gender.

    Returns dict with:
        prediction, probability,
        flipped_prediction, flipped_probability
    """
    row  = _build_row(user_input, feature_names, encoders)
    flip = _flip_gender(user_input, feature_names, encoders)

    prob       = model.predict_proba(row)[0, 1]
    pred       = int(prob >= 0.5)
    flip_prob  = model.predict_proba(flip)[0, 1]
    flip_pred  = int(flip_prob >= 0.5)

    return {
        "prediction":         pred,
        "probability":        float(prob),
        "flipped_prediction": flip_pred,
        "flipped_probability":float(flip_prob),
    }


def _build_row(user_input: dict, feature_names: list, encoders: dict) -> np.ndarray:
    """Convert user_input dict into a scaled feature vector."""
    merged = {**_DEFAULTS, **user_input}
    row    = []
    for col in feature_names:
        val = merged.get(col, _DEFAULTS.get(col, 0))
        if col in encoders and hasattr(encoders[col], "transform"):
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        row.append(float(val))

    row_arr = np.array([row])
    scaler  = encoders.get("__scaler__")
    num_idx = encoders.get("__num_indices__", [])
    if scaler is not None and len(num_idx) > 0:
        row_arr[:, num_idx] = scaler.transform(row_arr[:, num_idx])
    return row_arr


def _flip_gender(user_input: dict, feature_names: list, encoders: dict) -> np.ndarray:
    """Build the same row but with gender flipped."""
    flipped = user_input.copy()
    current = flipped.get("gender", "Male")
    flipped["gender"] = _GENDER_FLIP.get(current, "Female")
    return _build_row(flipped, feature_names, encoders)
