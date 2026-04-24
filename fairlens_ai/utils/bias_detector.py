"""
utils/bias_detector.py
Computes fairness metrics:
  - Demographic Parity Difference (DPD)
  - Equal Opportunity Difference (EOD)
  - Per-group statistics
  - An aggregated Bias Score (0–100)
"""

import numpy as np
from typing import Optional


def compute_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute fairness metrics for a binary classifier.

    Parameters
    ----------
    y_true      : true labels  (0 / 1)
    y_pred      : predicted labels (0 / 1)
    sensitive   : encoded sensitive attribute array (integers)

    Returns
    -------
    dict with keys:
        demographic_parity_diff, equal_opportunity_diff,
        group_stats, group_labels
    """
    metrics: dict = {}

    if sensitive is None or len(np.unique(sensitive)) < 2:
        # Can't compute group metrics without ≥ 2 groups
        metrics["demographic_parity_diff"] = 0.0
        metrics["equal_opportunity_diff"]  = 0.0
        metrics["group_stats"]             = {}
        return metrics

    groups = np.unique(sensitive)
    group_stats = {}

    positive_rates = {}
    tpr_rates      = {}

    for g in groups:
        mask = sensitive == g
        n    = mask.sum()
        if n == 0:
            continue

        pos_rate = y_pred[mask].mean()          # P(Ŷ=1 | group=g)
        positive_rates[g] = pos_rate

        # True Positive Rate (sensitivity / recall)
        actual_pos = y_true[mask] == 1
        if actual_pos.sum() > 0:
            tpr = y_pred[mask & (y_true == 1)].mean()
        else:
            tpr = 0.0
        tpr_rates[g] = tpr

        # Accuracy within group
        grp_acc = (y_pred[mask] == y_true[mask]).mean()

        group_stats[int(g)] = {
            "n":            int(n),
            "positive_rate": round(float(pos_rate), 4),
            "tpr":          round(float(tpr), 4),
            "accuracy":     round(float(grp_acc), 4),
        }

    # ── Demographic Parity Difference ────────────────────────────────────────
    # Defined as the maximum pairwise difference in positive prediction rates.
    rates = list(positive_rates.values())
    dpd = max(rates) - min(rates)
    # Sign: positive means first group has higher rate
    sorted_groups = sorted(positive_rates.keys(), key=lambda g: positive_rates[g], reverse=True)
    if len(sorted_groups) >= 2:
        dpd_signed = (positive_rates[sorted_groups[0]]
                      - positive_rates[sorted_groups[1]])
    else:
        dpd_signed = 0.0

    metrics["demographic_parity_diff"] = round(float(dpd_signed), 6)

    # ── Equal Opportunity Difference ─────────────────────────────────────────
    tpr_vals = list(tpr_rates.values())
    if len(tpr_vals) >= 2:
        sorted_tpr = sorted(tpr_rates.keys(), key=lambda g: tpr_rates[g], reverse=True)
        eod = tpr_rates[sorted_tpr[0]] - tpr_rates[sorted_tpr[1]]
    else:
        eod = 0.0

    metrics["equal_opportunity_diff"] = round(float(eod), 6)
    metrics["group_stats"]            = group_stats
    metrics["group_labels"]           = list(groups)

    return metrics


def compute_bias_score(metrics: dict) -> int:
    """
    Aggregate bias into a single 0–100 score.

    Formula (heuristic):
        score = min(100, (|DPD| * 300 + |EOD| * 200))

    Interpretation:
        0–20   : Very Fair ✅
        21–40  : Slight Bias 🟡
        41–60  : Moderate Bias 🟠
        61–80  : High Bias 🔴
        81–100 : Severe Bias ⛔
    """
    dpd = abs(metrics.get("demographic_parity_diff", 0))
    eod = abs(metrics.get("equal_opportunity_diff",  0))

    raw = dpd * 300 + eod * 200
    score = int(min(100, round(raw)))
    return score
