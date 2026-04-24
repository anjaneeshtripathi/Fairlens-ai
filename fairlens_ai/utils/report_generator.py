"""
utils/report_generator.py
Generates a plain-text / exportable fairness audit report.
"""

from datetime import datetime


def generate_report(
    bias_metrics:   dict,
    bias_score:     int,
    accuracy:       float,
    sensitive_attr: str,
    n_rows:         int,
) -> str:
    """
    Return a multi-line string suitable for download as a .txt report.
    """
    dpd = bias_metrics.get("demographic_parity_diff", 0)
    eod = bias_metrics.get("equal_opportunity_diff",  0)

    verdict_dpd = _verdict(abs(dpd))
    verdict_eod = _verdict(abs(eod))
    overall     = _overall_verdict(bias_score)

    group_stats = bias_metrics.get("group_stats", {})
    group_lines = ""
    for g, stats in group_stats.items():
        group_lines += (
            f"  Group {g}:\n"
            f"    n = {stats.get('n', 'N/A')}\n"
            f"    Positive Prediction Rate : {stats.get('positive_rate', 'N/A')}\n"
            f"    True Positive Rate (TPR) : {stats.get('tpr', 'N/A')}\n"
            f"    Accuracy                 : {stats.get('accuracy', 'N/A')}\n\n"
        )

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              FAIRLENS AI — FAIRNESS AUDIT REPORT                ║
╚══════════════════════════════════════════════════════════════════╝

Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset   : Adult Income Dataset (UCI)
Records   : {n_rows:,}
Model     : Logistic Regression

── SENSITIVE ATTRIBUTE ─────────────────────────────────────────────
Attribute analysed : {sensitive_attr.upper()}

── FAIRNESS METRICS ────────────────────────────────────────────────
Metric                        Value        Verdict
─────────────────────────────────────────────────────
Demographic Parity Difference : {dpd:+.6f}    {verdict_dpd}
Equal Opportunity Difference  : {eod:+.6f}    {verdict_eod}

Overall Bias Score            : {bias_score} / 100
Overall Verdict               : {overall}

── MODEL PERFORMANCE ───────────────────────────────────────────────
Overall Accuracy : {accuracy:.2%}

── GROUP-LEVEL STATISTICS ──────────────────────────────────────────
{group_lines}
── METRIC DEFINITIONS ──────────────────────────────────────────────
Demographic Parity Difference (DPD):
  DPD = P(Ŷ=1 | group A) − P(Ŷ=1 | group B)
  Ideal value: 0.  Threshold for concern: |DPD| > 0.05

Equal Opportunity Difference (EOD):
  EOD = TPR(group A) − TPR(group B)
  Measures whether the model is equally good at identifying
  positive cases across groups.
  Ideal value: 0.  Threshold for concern: |EOD| > 0.05

Bias Score:
  Aggregated metric on a 0–100 scale.
  0–20: Very Fair | 21–40: Slight Bias | 41–60: Moderate |
  61–80: High Bias | 81–100: Severe Bias

── RECOMMENDATIONS ─────────────────────────────────────────────────
{'✅ The model appears reasonably fair. Continue monitoring.' if bias_score <= 20 else
 '⚠️  Consider applying Reweighting or Fairness-Constrained training.'
 if bias_score <= 50 else
 '🚨 High bias detected. Apply mitigation before deployment.\n'
 '   Options: Reweighting, Adversarial Debiasing, Fairlearn constraints.'}

── DISCLAIMER ──────────────────────────────────────────────────────
This report is generated automatically by FairLens AI for
educational and hackathon purposes. It should not be used as the
sole basis for real-world deployment decisions.

══════════════════════════════════════════════════════════════════════
"""
    return report.strip()


def _verdict(abs_val: float) -> str:
    if abs_val <= 0.05:  return "✅ FAIR"
    if abs_val <= 0.10:  return "⚠️  SLIGHT BIAS"
    if abs_val <= 0.20:  return "🟠 MODERATE BIAS"
    return "🔴 HIGH BIAS"


def _overall_verdict(score: int) -> str:
    if score <= 20:  return "✅ VERY FAIR"
    if score <= 40:  return "🟡 SLIGHT BIAS"
    if score <= 60:  return "🟠 MODERATE BIAS"
    if score <= 80:  return "🔴 HIGH BIAS"
    return "⛔ SEVERE BIAS"
