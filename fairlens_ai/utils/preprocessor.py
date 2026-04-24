"""
utils/preprocessor.py
Cleans and encodes the Adult Income Dataset.
Returns train/test splits plus encoders for the What-If simulator.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Columns we actually use (drop leaky / irrelevant ones)
CATEGORICAL_COLS = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race", "gender", "native-country"
]
NUMERICAL_COLS = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]
TARGET_COL = "income"


def preprocess_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled)
    y_train, y_test : np.ndarray  (0/1)
    feature_names   : list[str]
    encoders        : dict  {col_name: LabelEncoder}
    """
    df = df.copy()

    # ── 1. Target encoding ──────────────────────────────────────────────────
    df["label"] = (df[TARGET_COL].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)

    # ── 2. Keep only relevant columns ───────────────────────────────────────
    use_cols = [c for c in CATEGORICAL_COLS if c in df.columns] + \
               [c for c in NUMERICAL_COLS   if c in df.columns]
    df = df[use_cols + ["label"]].copy()

    # ── 3. Encode categoricals ──────────────────────────────────────────────
    encoders: dict = {}
    for col in [c for c in CATEGORICAL_COLS if c in df.columns]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── 4. Feature matrix & target ──────────────────────────────────────────
    feature_names = [c for c in use_cols if c in df.columns]
    X = df[feature_names].values.astype(float)
    y = df["label"].values

    # ── 5. Train/test split (stratified) ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── 6. Scale numerical features ─────────────────────────────────────────
    num_indices = [feature_names.index(c) for c in NUMERICAL_COLS if c in feature_names]
    scaler = StandardScaler()
    X_train[:, num_indices] = scaler.fit_transform(X_train[:, num_indices])
    X_test[:, num_indices]  = scaler.transform(X_test[:, num_indices])
    encoders["__scaler__"]      = scaler
    encoders["__num_indices__"] = num_indices
    encoders["__feature_names__"] = feature_names

    return X_train, X_test, y_train, y_test, feature_names, encoders
