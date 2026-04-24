"""
utils/visualizer.py
Rich, publication-quality visualizations for FairLens AI.
Every function returns a matplotlib Figure on a dark theme.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Optional

# ── Dark theme applied globally ────────────────────────────────────────────────
BG       = "#0e0e1f"
SURFACE  = "#13132a"
BORDER   = "#2a2a4a"
TEXT     = "#c8c8e8"
MUTED    = "#6868a0"
PALETTE  = ["#6c63ff", "#3ecfcf", "#ff6b6b", "#ffd43b", "#51cf66", "#ffa94d"]
GREEN    = "#40c057"
RED      = "#ff6b6b"
AMBER    = "#ffa94d"

def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor"   : BG,
        "axes.facecolor"     : SURFACE,
        "axes.edgecolor"     : BORDER,
        "axes.labelcolor"    : MUTED,
        "axes.titlecolor"    : TEXT,
        "xtick.color"        : MUTED,
        "ytick.color"        : MUTED,
        "text.color"         : TEXT,
        "grid.color"         : BORDER,
        "grid.linestyle"     : "--",
        "grid.alpha"         : 0.5,
        "legend.facecolor"   : SURFACE,
        "legend.edgecolor"   : BORDER,
        "legend.labelcolor"  : TEXT,
        "font.family"        : "DejaVu Sans",
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.spines.left"   : True,
        "axes.spines.bottom" : True,
    })

_apply_theme()


# ══════════════════════════════════════════════════════════════════════════════
# 1. APPROVAL RATE BARS  (with error bands + fairness zone)
# ══════════════════════════════════════════════════════════════════════════════
def plot_approval_rates(y_pred, sensitive, sensitive_attr="gender"):
    """Horizontal bar chart with fairness threshold band."""
    groups = np.unique(sensitive)
    rates  = [float(y_pred[sensitive == g].mean()) for g in groups]
    labels = [str(g) for g in groups]
    avg    = np.mean(rates)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(groups) * 1.0)))
    colors  = [GREEN if abs(r - avg) <= 0.05 else RED for r in rates]
    bars    = ax.barh(labels, rates, color=colors, height=0.5,
                      edgecolor=BG, linewidth=1.5)

    # Fairness zone shading (±5 pp around average)
    ax.axvspan(avg - 0.05, avg + 0.05, alpha=0.12, color=GREEN, label="Fair zone (±5pp)")
    ax.axvline(avg, color=GREEN, linestyle="--", linewidth=1.4, alpha=0.8,
               label=f"Average: {avg:.1%}")

    for bar, rate, color in zip(bars, rates, colors):
        ax.text(rate + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", fontsize=11,
                color=TEXT, fontweight="bold")

    ax.set_xlim(0, max(rates) * 1.35)
    ax.set_xlabel("P(income > 50K)", labelpad=8)
    ax.set_title(f"Approval rate by {sensitive_attr}", pad=12, fontsize=13)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIDE-BY-SIDE CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(y_true, y_pred, sensitive, sensitive_attr="gender"):
    groups  = np.unique(sensitive)
    n       = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        mask = sensitive == g
        cm   = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        # Annotate with count AND percentage
        total = cm.sum()
        annot = np.array([[f"{v}\n({v/total:.1%})" for v in row] for row in cm])
        sns.heatmap(cm, annot=annot, fmt="", cmap=sns.dark_palette("#6c63ff", as_cmap=True),
                    xticklabels=["≤50K", ">50K"], yticklabels=["≤50K", ">50K"],
                    ax=ax, cbar=False, linewidths=2, linecolor=BG,
                    annot_kws={"size": 11, "color": "white"})
        grp_acc = (y_pred[mask] == y_true[mask]).mean()
        ax.set_title(f"{sensitive_attr} = {g}\n(acc: {grp_acc:.1%})", fontsize=11, pad=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    fig.suptitle("Confusion matrices by group", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROBABILITY DENSITY RIDGE PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_score_distributions(y_pred_proba, sensitive, sensitive_attr="gender"):
    """Overlapping KDE curves showing score distributions per group."""
    from scipy.stats import gaussian_kde

    groups = np.unique(sensitive)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.linspace(0, 1, 300)

    for i, g in enumerate(groups):
        probs = y_pred_proba[sensitive == g]
        kde   = gaussian_kde(probs, bw_method=0.12)
        y     = kde(x)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, color=color, linewidth=2.5, label=f"{sensitive_attr}={g} (n={len(probs)})")
        ax.fill_between(x, y, alpha=0.15, color=color)

    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.2,
               label="Decision boundary (0.5)", alpha=0.7)
    ax.set_xlabel("Predicted probability P(income > 50K)", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title("Prediction score distribution by group", fontsize=13, pad=12)
    ax.legend(fontsize=10); ax.yaxis.grid(True); ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. PER-GROUP ROC CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_by_group(y_true, y_pred_proba, sensitive, sensitive_attr="gender"):
    """Separate ROC curve per group — reveals per-group discrimination power."""
    groups = np.unique(sensitive)
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for i, g in enumerate(groups):
        mask            = sensitive == g
        fpr, tpr, _     = roc_curve(y_true[mask], y_pred_proba[mask])
        roc_auc_val     = auc(fpr, tpr)
        color           = PALETTE[i % len(PALETTE)]
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{sensitive_attr}={g}  AUC={roc_auc_val:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax.plot([0, 1], [0, 1], color=BORDER, linestyle="--", linewidth=1.2, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", labelpad=8)
    ax.set_ylabel("True Positive Rate", labelpad=8)
    ax.set_title("ROC curve by group", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.xaxis.grid(True); ax.yaxis.grid(True)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. CALIBRATION PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_calibration(y_true, y_pred_proba, sensitive, sensitive_attr="gender"):
    """Reliability diagram — are high-confidence predictions actually correct?"""
    groups  = np.unique(sensitive)
    n_bins  = 10
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Perfect calibration")

    for i, g in enumerate(groups):
        mask  = sensitive == g
        probs = y_pred_proba[mask]
        true  = y_true[mask]
        bins  = np.linspace(0, 1, n_bins + 1)
        frac_pos, mean_pred = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (probs >= lo) & (probs < hi)
            if in_bin.sum() >= 5:
                frac_pos.append(true[in_bin].mean())
                mean_pred.append(probs[in_bin].mean())

        color = PALETTE[i % len(PALETTE)]
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2,
                markersize=7, label=f"{sensitive_attr}={g}")

    ax.set_xlabel("Mean predicted probability", labelpad=8)
    ax.set_ylabel("Fraction of positives", labelpad=8)
    ax.set_title("Calibration plot by group", fontsize=13, pad=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10); ax.xaxis.grid(True); ax.yaxis.grid(True)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE — DIVERGING BARS
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names):
    """Diverging horizontal bar chart of LR coefficients, sensitive features highlighted."""
    SENSITIVE = {"gender", "race", "sex", "native-country", "relationship"}
    coef_df   = sorted(zip(feature_names, model.coef_[0]), key=lambda x: x[1])
    names, coefs = zip(*coef_df)

    colors = []
    for name, c in zip(names, coefs):
        if name in SENSITIVE:
            colors.append(AMBER)
        elif c > 0:
            colors.append(GREEN)
        else:
            colors.append(RED)

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.44)))
    bars = ax.barh(names, coefs, color=colors, height=0.65,
                   edgecolor=BG, linewidth=1)
    ax.axvline(0, color=BORDER, linewidth=1.2)

    for bar, val in zip(bars, coefs):
        xpos = val + (0.01 if val >= 0 else -0.01)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9, color=TEXT)

    legend_handles = [
        mpatches.Patch(color=GREEN,  label="→ Increases >50K prediction"),
        mpatches.Patch(color=RED,    label="→ Decreases >50K prediction"),
        mpatches.Patch(color=AMBER,  label="Sensitive attribute (watch for bias)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
    ax.set_xlabel("Logistic regression coefficient", labelpad=8)
    ax.set_title("Feature importance — model coefficients", fontsize=13, pad=12)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. FAIRNESS RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_fairness_radar(metrics_dict: dict):
    """
    Radar / spider chart comparing multiple fairness dimensions.
    metrics_dict: {label: {dpd, eod, accuracy, ...}}
    """
    dims = ["DPD (inv)", "EOD (inv)", "Accuracy", "TPR parity", "FPR parity"]
    N    = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_color(BORDER)
    ax.grid(color=BORDER, linewidth=0.8)

    for i, (label, vals) in enumerate(metrics_dict.items()):
        values = [
            1 - min(1, abs(vals.get("dpd", 0)) * 5),
            1 - min(1, abs(vals.get("eod", 0)) * 5),
            vals.get("accuracy", 0.8),
            1 - min(1, abs(vals.get("tpr_diff", 0)) * 5),
            1 - min(1, abs(vals.get("fpr_diff", 0)) * 5),
        ]
        values += values[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10, color=TEXT)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.set_title("Fairness radar", fontsize=13, pad=20, color=TEXT)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. BIAS MITIGATION BEFORE/AFTER  (improved layout)
# ══════════════════════════════════════════════════════════════════════════════
def plot_bias_comparison(orig_metrics, new_metrics, orig_acc, new_acc):
    """2×2 grid: bias score, DPD, EOD, accuracy — before vs after."""
    from utils.bias_detector import compute_bias_score
    orig_score = compute_bias_score(orig_metrics)
    new_score  = compute_bias_score(new_metrics)

    items = [
        ("Bias score\n(lower = fairer)",  orig_score,                                  new_score,                                 "lower"),
        ("|DPD|\n(lower = fairer)",       abs(orig_metrics.get("demographic_parity_diff",0))*100, abs(new_metrics.get("demographic_parity_diff",0))*100, "lower"),
        ("|EOD|\n(lower = fairer)",       abs(orig_metrics.get("equal_opportunity_diff",0))*100,  abs(new_metrics.get("equal_opportunity_diff",0))*100,  "lower"),
        ("Accuracy %\n(higher = better)", orig_acc * 100,                               new_acc * 100,                             "higher"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, (label, before, after, direction) in zip(axes, items):
        improved = (after < before) if direction == "lower" else (after > before)
        after_color  = GREEN if improved else RED
        bars = ax.bar(["Before", "After"], [before, after],
                      color=[RED if direction == "lower" else MUTED, after_color],
                      edgecolor=BG, linewidth=1.5, width=0.5)
        for bar, v in zip(bars, [before, after]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(before, after) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=11, color=TEXT, fontweight="bold")

        delta = after - before
        sign  = "+" if delta > 0 else ""
        delta_color = GREEN if improved else RED
        ax.set_title(f"{label}\n{sign}{delta:.1f}", fontsize=10, pad=8, color=delta_color)
        ax.set_ylim(0, max(before, after) * 1.28)
        ax.yaxis.grid(True); ax.set_axisbelow(True)

    fig.suptitle("Bias mitigation: before vs after", fontsize=14, y=1.02, color=TEXT)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9. INTERSECTIONAL HEATMAP  (gender × race approval rates)
# ══════════════════════════════════════════════════════════════════════════════
def plot_intersectional_heatmap(y_pred, df_test_raw):
    """
    Heatmap of approval rates across gender × race.
    df_test_raw: raw (un-encoded) test-set DataFrame with 'gender' and 'race' columns.
    """
    if df_test_raw is None or "gender" not in df_test_raw.columns:
        return None

    pivot = df_test_raw.copy()
    pivot["predicted"] = y_pred
    matrix = pivot.groupby(["gender", "race"])["predicted"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        matrix, annot=True, fmt=".1%",
        cmap=sns.diverging_palette(10, 133, as_cmap=True),
        center=matrix.values[~np.isnan(matrix.values)].mean(),
        ax=ax, linewidths=1.5, linecolor=BG,
        cbar_kws={"label": "Approval rate"},
        annot_kws={"size": 10, "color": "white", "weight": "bold"},
    )
    ax.set_title("Intersectional approval rates: gender × race", fontsize=13, pad=12)
    ax.set_xlabel("Race", labelpad=8); ax.set_ylabel("Gender", labelpad=8)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 10. SHAP SUMMARY (if shap installed)
# ══════════════════════════════════════════════════════════════════════════════
def plot_shap_summary(model, X_test, feature_names):
    import shap
    explainer   = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test[:300])

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(
        shap_values, X_test[:300],
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        color_bar=True,
    )
    plt.title("SHAP feature importance (dot plot)", fontsize=13, pad=12)
    plt.tight_layout()
    return plt.gcf()


# ══════════════════════════════════════════════════════════════════════════════
# 11. WHAT-IF PROBABILITY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def plot_whatif_comparison(gender_sel, flip_lbl, prob_orig, prob_flip):
    """Horizontal bar chart comparing original vs gender-flipped probability."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    labels  = [gender_sel, f"{flip_lbl} (flipped)"]
    probs   = [prob_orig, prob_flip]
    colors  = [PALETTE[0], PALETTE[1]]
    bars    = ax.barh(labels, probs, color=colors, height=0.4,
                      edgecolor=BG, linewidth=1.5)

    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.2,
               label="Decision boundary (0.5)", alpha=0.7)
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=11,
                color=TEXT, fontweight="bold")

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("P(income > 50K)", labelpad=8)
    ax.set_title("Probability: original vs gender-flipped", pad=10, fontsize=12)
    ax.legend(fontsize=9); ax.xaxis.grid(True)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 12. BIAS SCORE OVER DECISION THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════
def plot_threshold_bias_tradeoff(y_true, y_pred_proba, sensitive):
    """
    Shows how bias score and accuracy change as you sweep the decision threshold.
    Useful to find the threshold that optimises the fairness-accuracy tradeoff.
    """
    from utils.bias_detector import compute_bias_metrics, compute_bias_score
    from sklearn.metrics import accuracy_score

    thresholds  = np.linspace(0.1, 0.9, 80)
    bias_scores = []
    accuracies  = []

    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        m        = compute_bias_metrics(y_true, y_pred_t, sensitive)
        bias_scores.append(compute_bias_score(m))
        accuracies.append(accuracy_score(y_true, y_pred_t) * 100)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, bias_scores, color=RED, linewidth=2.5, label="Bias score")
    ax1.fill_between(thresholds, bias_scores, alpha=0.1, color=RED)
    ax2.plot(thresholds, accuracies, color=PALETTE[0], linewidth=2.5,
             linestyle="--", label="Accuracy %")

    ax1.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.2, label="Default threshold (0.5)")

    # Mark minimum bias point
    min_bias_idx = int(np.argmin(bias_scores))
    ax1.scatter([thresholds[min_bias_idx]], [bias_scores[min_bias_idx]],
                color=GREEN, s=80, zorder=5, label=f"Min bias @ t={thresholds[min_bias_idx]:.2f}")

    ax1.set_xlabel("Decision threshold", labelpad=8)
    ax1.set_ylabel("Bias score (0–100)", color=RED, labelpad=8)
    ax2.set_ylabel("Accuracy (%)", color=PALETTE[0], labelpad=8)
    ax1.set_title("Fairness–accuracy trade-off across thresholds", fontsize=13, pad=12)
    ax1.set_xlim(0.1, 0.9); ax1.set_ylim(0, 110)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax1.yaxis.grid(True)
    fig.tight_layout()
    return fig
