"""
utils/data_loader.py
Loads the Adult Income Dataset (UCI).
Falls back to a synthetic version if network is unavailable.
"""

import pandas as pd
import numpy as np
import os


COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adult.csv")


def load_adult_dataset() -> pd.DataFrame:
    """
    Load the Adult Income Dataset.
    Tries local cache → URL → synthetic fallback.
    """
    # 1. Try local cache
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return _clean(df)

    # 2. Try downloading from UCI
    try:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/adult/adult.data"
        )
        df = pd.read_csv(url, names=COLUMN_NAMES, na_values=" ?", skipinitialspace=True)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return _clean(df)
    except Exception:
        pass

    # 3. Synthetic fallback (reproducible for demo)
    return _make_synthetic()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from strings and standardise column names."""
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    # Normalise income label
    df["income"] = df["income"].str.replace(".", "", regex=False)
    # Drop fnlwgt (sampling weight, not a real feature)
    if "fnlwgt" in df.columns:
        df.drop(columns=["fnlwgt"], inplace=True)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


def _make_synthetic() -> pd.DataFrame:
    """
    Creates a synthetic dataset that mimics Adult Income biases
    so the app works even without internet.
    n = 10_000 rows.
    """
    np.random.seed(42)
    n = 10_000

    gender  = np.random.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    race    = np.random.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        size=n, p=[0.855, 0.095, 0.03, 0.01, 0.01]
    )
    age     = np.random.randint(18, 75, n)
    edu_num = np.random.randint(1, 17, n)
    hours   = np.random.randint(20, 70, n)
    cap_g   = np.random.choice([0, 5000, 10000, 15000, 50000], size=n,
                                p=[0.85, 0.05, 0.05, 0.03, 0.02])
    cap_l   = np.random.choice([0, 1500, 2000], size=n, p=[0.9, 0.05, 0.05])

    workclass  = np.random.choice(
        ["Private", "Self-emp-not-inc", "Self-emp-inc",
         "Federal-gov", "Local-gov", "State-gov"],
        size=n, p=[0.7, 0.08, 0.05, 0.03, 0.07, 0.07]
    )
    occupation = np.random.choice(
        ["Prof-specialty", "Craft-repair", "Exec-managerial",
         "Adm-clerical", "Sales", "Other-service"],
        size=n
    )
    education = np.random.choice(
        ["Bachelors", "Some-college", "11th", "HS-grad",
         "Prof-school", "Assoc-acdm", "Assoc-voc",
         "9th", "7th-8th", "12th", "Masters", "Doctorate"],
        size=n
    )
    marital = np.random.choice(
        ["Married-civ-spouse", "Divorced", "Never-married",
         "Separated", "Widowed"],
        size=n, p=[0.45, 0.15, 0.28, 0.07, 0.05]
    )
    relationship = np.random.choice(
        ["Wife", "Own-child", "Husband", "Not-in-family",
         "Other-relative", "Unmarried"],
        size=n
    )
    country = np.array(["United-States"] * n)

    # Build biased income label (males more likely, higher edu more likely)
    logit = (
        -3.0
        + 0.03 * age
        + 0.20 * edu_num
        + 0.02 * hours
        + 0.8  * (gender == "Male").astype(float)      # gender bias
        + 0.3  * (race == "White").astype(float)       # race bias
        + cap_g / 20000
        - cap_l / 5000
    )
    prob   = 1 / (1 + np.exp(-logit))
    income = np.where(
        np.random.uniform(size=n) < prob, ">50K", "<=50K"
    )

    df = pd.DataFrame({
        "age": age, "workclass": workclass, "education": education,
        "education-num": edu_num, "marital-status": marital,
        "occupation": occupation, "relationship": relationship,
        "race": race, "gender": gender,
        "capital-gain": cap_g, "capital-loss": cap_l,
        "hours-per-week": hours, "native-country": country,
        "income": income
    })
    return df
