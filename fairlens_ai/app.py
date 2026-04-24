"""
FairLens AI ─ Streamlit Application  (v3 + Gemini AI Advisor)
==============================================================
Run:  streamlit run app.py
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os

# ── third-party ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score
)

# ── local modules ──────────────────────────────────────────────────────────────
from utils.data_loader      import load_adult_dataset
from utils.preprocessor     import preprocess_data
from utils.model_trainer    import train_model, evaluate_model
from utils.bias_detector    import compute_bias_metrics, compute_bias_score
from utils.whatif_simulator import predict_whatif
from utils.mitigator        import apply_reweighting, apply_remove_sensitive
from utils.report_generator import generate_report
from utils.gemini_advisor   import ask_gemini, build_bias_context   # ← NEW
from utils.visualizer       import (
    plot_approval_rates, plot_confusion_matrices, plot_score_distributions,
    plot_roc_by_group, plot_calibration, plot_feature_importance,
    plot_fairness_radar, plot_bias_comparison, plot_intersectional_heatmap,
    plot_threshold_bias_tradeoff, plot_whatif_comparison,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "FairLens AI",
    page_icon   = "⚖️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e0e1f 0%, #141428 100%);
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }
[data-testid="stSidebar"] hr { border-color: #2a2a4a; }

.fl-card {
    background: linear-gradient(135deg, #13132a 0%, #1c1c38 100%);
    border: 1px solid #2e2e5a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
.fl-card-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:#7878aa; margin-bottom:0.35rem; }
.fl-card-value { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; line-height:1; }
.fl-card-sub   { font-size:0.72rem; color:#5a5a88; margin-top:0.3rem; }

.score-severe   { color: #ff3860; }
.score-high     { color: #ff6b6b; }
.score-moderate { color: #ffa94d; }
.score-slight   { color: #ffd43b; }
.score-fair     { color: #40c057; }

.fl-section-title {
    font-family:'Space Mono',monospace; font-size:0.9rem; font-weight:700;
    letter-spacing:0.05em; color:#a0a0cc; text-transform:uppercase;
    margin:1.4rem 0 0.8rem; padding-bottom:0.4rem;
    border-bottom:1px solid #2a2a4a;
}

.fl-alert-danger  { background:#2d0a0a; border:1px solid #6b1a1a; border-left:4px solid #ff3860; border-radius:8px; padding:0.9rem 1.1rem; color:#ff8fa3; font-size:0.9rem; margin:0.8rem 0; }
.fl-alert-success { background:#0a2d1a; border:1px solid #1a6b3a; border-left:4px solid #40c057; border-radius:8px; padding:0.9rem 1.1rem; color:#69db7c; font-size:0.9rem; margin:0.8rem 0; }
.fl-alert-info    { background:#0a1a2d; border:1px solid #1a3a6b; border-left:4px solid #339af0; border-radius:8px; padding:0.9rem 1.1rem; color:#74c0fc; font-size:0.9rem; margin:0.8rem 0; }
.fl-alert-warning { background:#2d1f0a; border:1px solid #6b4a1a; border-left:4px solid #ffa94d; border-radius:8px; padding:0.9rem 1.1rem; color:#ffd8a8; font-size:0.9rem; margin:0.8rem 0; }

.pred-badge-pos { display:inline-block; background:#0d3320; border:1px solid #2b6b43; color:#51cf66; border-radius:999px; padding:0.35rem 1rem; font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; }
.pred-badge-neg { display:inline-block; background:#2d0a0a; border:1px solid #6b1a1a; color:#ff6b6b; border-radius:999px; padding:0.35rem 1rem; font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; }

.prob-bar-wrap { background:#1a1a2e; border-radius:999px; height:12px; overflow:hidden; margin:0.4rem 0; }
.prob-bar-fill { height:100%; border-radius:999px; }

.pill { display:inline-block; border-radius:999px; padding:0.2rem 0.7rem; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
.pill-red    { background:#3d0a0a; color:#ff6b6b; border:1px solid #6b1a1a; }
.pill-green  { background:#0a2d18; color:#51cf66; border:1px solid #1a6b38; }
.pill-yellow { background:#2d2000; color:#ffd43b; border:1px solid #6b5000; }
.pill-orange { background:#2d1500; color:#ffa94d; border:1px solid #6b3800; }

.hero-banner {
    background:linear-gradient(120deg,#0e0e20 0%,#1a1435 50%,#0e1a20 100%);
    border:1px solid #2a2a4a; border-radius:16px; padding:2rem 2.5rem;
    margin-bottom:1.5rem; position:relative; overflow:hidden;
}
.hero-title {
    font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700;
    background:linear-gradient(90deg,#a78bfa,#60d9f9,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem;
}
.hero-sub  { color:#7878aa; font-size:0.95rem; max-width:640px; line-height:1.6; }
.hero-tag  { display:inline-block; background:#1a1435; border:1px solid #3d2e80; color:#a78bfa; border-radius:6px; padding:0.2rem 0.65rem; font-size:0.75rem; font-weight:500; margin:0.6rem 0.3rem 0 0; }

/* Gemini chat bubbles */
.gemini-user { background:#1a1435; border:1px solid #3d2e80; border-radius:12px 12px 2px 12px; padding:0.8rem 1.1rem; color:#c8c8e8; font-size:0.9rem; margin:0.5rem 0; }
.gemini-ai   { background:#0a2d1a; border:1px solid #1a6b3a; border-radius:12px 12px 12px 2px; padding:0.8rem 1.1rem; color:#d0ffd0; font-size:0.9rem; margin:0.5rem 0; line-height:1.7; }
.gemini-hdr  { font-family:'Space Mono',monospace; font-size:0.75rem; color:#5a5a88; margin-bottom:0.3rem; }

.stButton > button { background:linear-gradient(90deg,#6c63ff 0%,#3ecfcf 100%) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:600 !important; padding:0.55rem 1.6rem !important; width:100% !important; }
.stButton > button:hover { opacity:0.85 !important; }

.stProgress > div > div { background:linear-gradient(90deg,#6c63ff,#3ecfcf) !important; }
hr { border-color:#2a2a4a !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def score_colour_class(s):
    if s > 80: return "score-severe"
    if s > 60: return "score-high"
    if s > 40: return "score-moderate"
    if s > 20: return "score-slight"
    return "score-fair"

def score_emoji(s):
    if s > 80: return "⛔"
    if s > 60: return "🔴"
    if s > 40: return "🟠"
    if s > 20: return "🟡"
    return "✅"

def pill(text, style):
    return f'<span class="pill pill-{style}">{text}</span>'

def metric_card(label, value, sub="", colour="#a78bfa"):
    return f"""<div class="fl-card">
        <div class="fl-card-label">{label}</div>
        <div class="fl-card-value" style="color:{colour};">{value}</div>
        <div class="fl-card-sub">{sub}</div>
    </div>"""

def prob_bar(prob, color="#6c63ff"):
    pct = round(prob * 100, 1)
    return f"""<div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div></div>
    <div style="font-size:0.8rem;color:#7878aa;">{pct}%</div>"""

def dpd_verdict(dpd_abs):
    if dpd_abs <= 0.05: return pill("Fair ✅", "green")
    if dpd_abs <= 0.10: return pill("Slight Bias", "yellow")
    if dpd_abs <= 0.20: return pill("Moderate Bias", "orange")
    return pill("High Bias ⚠️", "red")

# ── Dark matplotlib theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#0e0e1f","axes.facecolor":"#13132a",
    "axes.edgecolor":"#2a2a4a","axes.labelcolor":"#a0a0cc",
    "xtick.color":"#6868a0","ytick.color":"#6868a0","text.color":"#c8c8e8",
    "grid.color":"#1e1e38","grid.linestyle":"--","grid.alpha":0.6,
    "legend.facecolor":"#13132a","legend.edgecolor":"#2a2a4a",
    "font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
})
PALETTE = ["#6c63ff","#3ecfcf","#ff6b6b","#ffd43b","#51cf66","#ffa94d"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL  (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⚙️ Loading dataset…")
def get_data():
    df = load_adult_dataset()
    X_train, X_test, y_train, y_test, feature_names, encoders = preprocess_data(df)
    return df, X_train, X_test, y_train, y_test, feature_names, encoders

@st.cache_resource(show_spinner="🤖 Training model…")
def get_model(_X_train, _y_train):
    return train_model(_X_train, _y_train)

df, X_train, X_test, y_train, y_test, feature_names, encoders = get_data()
model        = get_model(X_train, y_train)
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">⚖️</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.15rem;font-weight:700;color:#a78bfa;margin-top:0.3rem;">FairLens AI</div>
        <div style="font-size:0.72rem;color:#5a5a88;margin-top:0.25rem;">Bias Detection &amp; Fairness Analysis</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Dataset Explorer",
        "🤖  Model & Accuracy",
        "🔍  Bias Detection",
        "📈  Visualizations",
        "🧠  Explainable AI",
        "🔄  What-If Simulator",
        "🛡️  Bias Mitigation",
        "🤖  Gemini AI Advisor",   # ← NEW
        "📄  Fairness Report",
    ], label_visibility="collapsed")

    st.markdown("---")
    sensitive_attr = st.selectbox("Sensitive Attribute", ["gender", "race"],
        help="Choose which protected attribute to analyse.")

    sens_idx       = feature_names.index(sensitive_attr) if sensitive_attr in feature_names else 0
    sensitive_test = X_test[:, sens_idx]
    bias_metrics   = compute_bias_metrics(y_test, y_pred, sensitive_test)
    bias_score_val = compute_bias_score(bias_metrics)
    sc_class       = score_colour_class(bias_score_val)
    sc_emoji_str   = score_emoji(bias_score_val)
    sc_c           = "#ff3860" if bias_score_val > 60 else ("#ffa94d" if bias_score_val > 30 else "#40c057")

    st.markdown(f"""
    <div style="text-align:center;margin-top:0.5rem;">
        <div style="font-size:0.7rem;color:#5a5a88;text-transform:uppercase;letter-spacing:0.08em;">Current Bias Score</div>
        <div class="{sc_class}" style="font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;">{bias_score_val}</div>
        <div style="font-size:0.75rem;color:#5a5a88;">out of 100 {sc_emoji_str}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#3a3a5a;text-align:center;">Powered by Google Gemini ✨<br/>Adult Income Dataset · UCI</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚖️ FairLens AI</div>
        <div class="hero-sub">Detect bias in ML models, explain decisions, simulate what-if scenarios, and apply mitigation techniques — all in one place. Now powered by Google Gemini.</div>
        <div>
            <span class="hero-tag">🔍 Bias Detection</span>
            <span class="hero-tag">🧠 Explainable AI</span>
            <span class="hero-tag">🔄 What-If Simulator</span>
            <span class="hero-tag">🛡️ Bias Mitigation</span>
            <span class="hero-tag">✨ Gemini AI Advisor</span>
            <span class="hero-tag">📄 Fairness Report</span>
        </div>
    </div>""", unsafe_allow_html=True)

    acc = evaluate_model(model, X_test, y_test)
    dpd = bias_metrics.get("demographic_parity_diff", 0)
    dpd_color = "#ff6b6b" if abs(dpd) > 0.1 else ("#ffd43b" if abs(dpd) > 0.05 else "#40c057")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Dataset Size",          f"{len(df):,}",    "rows · Adult Income UCI",           "#a78bfa"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Model Accuracy",        f"{acc:.1%}",      "Logistic Regression · test set",    "#3ecfcf"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Demographic Parity Gap",f"{dpd:+.3f}",     f"Sensitive: {sensitive_attr}",      dpd_color), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Bias Score",            f"{bias_score_val}/100", sc_emoji_str + " " + (
        "Very Fair" if bias_score_val<=20 else "Slight Bias" if bias_score_val<=40 else
        "Moderate"  if bias_score_val<=60 else "High Bias"   if bias_score_val<=80 else "Severe"
    ), sc_c), unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="fl-section-title">Approval Rate by Group</div>', unsafe_allow_html=True)
        groups = np.unique(sensitive_test)
        rates  = [float(y_pred[sensitive_test == g].mean()) for g in groups]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar([str(g) for g in groups], rates, color=PALETTE[:len(groups)],
                      width=0.5, edgecolor="#0e0e1f", linewidth=1.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                    f"{rate:.1%}", ha="center", va="bottom", fontsize=11, color="#c8c8e8", fontweight="bold")
        ax.axhline(np.mean(rates), color="#4a4a7a", linestyle="--", linewidth=1.2, label=f"Average: {np.mean(rates):.1%}")
        ax.set_ylabel("P( income > 50K )"); ax.set_xlabel(sensitive_attr.capitalize())
        ax.set_ylim(0, max(rates)*1.3); ax.legend(fontsize=9); ax.yaxis.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_right:
        st.markdown('<div class="fl-section-title">Bias Score Gauge</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"aspect":"equal"})
        fig.patch.set_facecolor("#0e0e1f"); ax.set_facecolor("#0e0e1f")
        theta = np.linspace(np.pi, 0, 300)
        ax.plot(np.cos(theta), np.sin(theta), color="#2a2a4a", linewidth=18)
        for seg, clr in [
            (np.linspace(np.pi,     np.pi*0.8, 60), "#40c057"),
            (np.linspace(np.pi*0.8, np.pi*0.6, 60), "#ffd43b"),
            (np.linspace(np.pi*0.6, np.pi*0.4, 60), "#ffa94d"),
            (np.linspace(np.pi*0.4, np.pi*0.2, 60), "#ff6b6b"),
            (np.linspace(np.pi*0.2, 0,          60), "#ff3860"),
        ]:
            ax.plot(np.cos(seg), np.sin(seg), color=clr, linewidth=18, alpha=0.9)
        needle_angle = np.pi - (bias_score_val/100)*np.pi
        ax.annotate("", xy=(0.78*np.cos(needle_angle), 0.78*np.sin(needle_angle)),
                    xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=20))
        ax.plot(0, 0, "o", color="white", markersize=10, zorder=5)
        ax.text(0, -0.22, str(bias_score_val), ha="center", va="center",
                fontsize=36, fontweight="bold", color=sc_c, fontfamily="monospace")
        ax.text(0, -0.45, "Bias Score", ha="center", va="center", fontsize=11, color="#6868a0")
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.6,1.1); ax.axis("off")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="fl-section-title">How to Use FairLens AI</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (icon, title, desc) in zip([c1,c2,c3,c4,c5], [
        ("📊","1. Explore Data",  "Browse the dataset and check sensitive attribute breakdowns."),
        ("🔍","2. Detect Bias",   "Measure Demographic Parity and Equal Opportunity across groups."),
        ("🔄","3. Simulate",      "Flip sensitive attributes in the What-If simulator."),
        ("🛡️","4. Mitigate",     "Apply reweighting and compare bias before vs after."),
        ("✨","5. Ask Gemini",    "Get AI-powered recommendations from Google Gemini."),
    ]):
        with col:
            st.markdown(f"""<div class="fl-card" style="min-height:130px;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:600;color:#c8c8e8;font-size:0.9rem;margin-bottom:0.35rem;">{title}</div>
                <div style="font-size:0.8rem;color:#6868a0;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Explorer":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">📊 Dataset Explorer</h2>', unsafe_allow_html=True)
    st.caption("Adult Income Dataset · UCI Machine Learning Repository")

    pos_rate = (df["income"] == ">50K").mean()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(metric_card("Rows",          f"{len(df):,}",      "total records",         "#a78bfa"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Features",      f"{df.shape[1]-1}",  "input columns",         "#3ecfcf"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Target",        "income >50K",       "binary classification", "#ffd43b"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Positive Rate", f"{pos_rate:.1%}",   ">50K earners",          "#ff6b6b" if pos_rate<0.35 else "#40c057"), unsafe_allow_html=True)

    st.markdown('<div class="fl-section-title">Raw Data Preview</div>', unsafe_allow_html=True)
    n_rows = st.slider("Rows to display", 10, 500, 50, 10)
    st.dataframe(df.head(n_rows), use_container_width=True, height=320)

    st.markdown('<div class="fl-section-title">Distributions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        vc = df["income"].value_counts()
        ax.pie(vc, labels=vc.index, colors=["#6c63ff","#3ecfcf"],
               autopct="%1.1f%%", startangle=90,
               textprops={"color":"#c8c8e8","fontsize":10},
               wedgeprops={"edgecolor":"#0e0e1f","linewidth":2})
        ax.set_title("Income Distribution", color="#c8c8e8", fontsize=11, pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        g_inc = df.groupby(["gender","income"]).size().unstack(fill_value=0)
        g_inc.plot(kind="bar", ax=ax, color=["#ff6b6b","#40c057"], edgecolor="#0e0e1f", linewidth=1.5)
        ax.set_title("Income by Gender", color="#c8c8e8", fontsize=11)
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=9); ax.yaxis.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col3:
        fig, ax = plt.subplots(figsize=(4.5,3.2))
        df[df["income"]==">50K"]["age"].hist(ax=ax, bins=25, color="#6c63ff", alpha=0.75, label=">50K", edgecolor="#0e0e1f")
        df[df["income"]!= ">50K"]["age"].hist(ax=ax, bins=25, color="#3ecfcf", alpha=0.55, label="≤50K",edgecolor="#0e0e1f")
        ax.set_title("Age Distribution by Income", color="#c8c8e8", fontsize=11)
        ax.set_xlabel("Age"); ax.legend(fontsize=9); ax.yaxis.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="fl-section-title">Sensitive Attribute Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gender**")
        gdf = df.groupby("gender").apply(lambda x: pd.Series({
            "Count": len(x), "% Dataset": f"{len(x)/len(df):.1%}", ">50K Rate": f"{(x['income']=='>50K').mean():.1%}"})).reset_index()
        st.dataframe(gdf, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Race**")
        rdf = df.groupby("race").apply(lambda x: pd.Series({
            "Count": len(x), "% Dataset": f"{len(x)/len(df):.1%}", ">50K Rate": f"{(x['income']=='>50K').mean():.1%}"})).reset_index()
        st.dataframe(rdf, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL & ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model & Accuracy":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">🤖 Model & Accuracy</h2>', unsafe_allow_html=True)
    acc         = evaluate_model(model, X_test, y_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(metric_card("Accuracy",  f"{acc:.2%}",                             "overall test set"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Precision", f"{report_dict['1']['precision']:.2%}",   "class >50K"),       unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Recall",    f"{report_dict['1']['recall']:.2%}",      "class >50K"),       unsafe_allow_html=True)
    with c4: st.markdown(metric_card("F1 Score",  f"{report_dict['1']['f1-score']:.2%}",    "class >50K"),       unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="fl-section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5,3.8))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.dark_palette("#6c63ff", as_cmap=True),
                    xticklabels=["≤50K",">50K"], yticklabels=["≤50K",">50K"],
                    ax=ax, cbar=False, linewidths=2, linecolor="#0e0e1f",
                    annot_kws={"size":16,"weight":"bold","color":"white"})
        ax.set_xlabel("Predicted", labelpad=8); ax.set_ylabel("Actual", labelpad=8)
        ax.set_title("Confusion Matrix", pad=12)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="fl-section-title">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4.5,3.8))
        ax.plot(fpr, tpr, color="#6c63ff", linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.12, color="#6c63ff")
        ax.plot([0,1],[0,1], color="#2a2a4a", linestyle="--", linewidth=1.2)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", pad=12); ax.legend(fontsize=10)
        ax.yaxis.grid(True); ax.xaxis.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="fl-section-title">Model Card</div>', unsafe_allow_html=True)
    st.markdown("""
    | Parameter | Value |
    |---|---|
    | Algorithm | Logistic Regression |
    | Solver | lbfgs |
    | Max Iterations | 1000 |
    | Class Weight | balanced |
    | Train / Test Split | 80% / 20% stratified |
    | Feature Scaling | StandardScaler |
    | Categorical Encoding | LabelEncoder |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BIAS DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Bias Detection":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">🔍 Bias Detection</h2>', unsafe_allow_html=True)

    dpd         = bias_metrics.get("demographic_parity_diff", 0)
    eod         = bias_metrics.get("equal_opportunity_diff",  0)
    group_stats = bias_metrics.get("group_stats", {})

    dpd_c = "#ff6b6b" if abs(dpd)>0.1 else ("#ffd43b" if abs(dpd)>0.05 else "#40c057")
    eod_c = "#ff6b6b" if abs(eod)>0.1 else ("#ffd43b" if abs(eod)>0.05 else "#40c057")

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(metric_card("Demographic Parity Diff", f"{dpd:+.4f}", "Ideal: 0 · Threshold: |0.05|", dpd_c), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Equal Opportunity Diff",  f"{eod:+.4f}", "TPR gap across groups",        eod_c), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Bias Score", f"{bias_score_val}/100", sc_emoji_str + " Composite metric", sc_c), unsafe_allow_html=True)

    with st.expander("📐 Demographic Parity Difference (DPD)", expanded=True):
        st.markdown(f"""
**Formula:** `DPD = P(Ŷ=1 | group A) − P(Ŷ=1 | group B)`

| DPD Value | Interpretation |
|---|---|
| 0.000–0.050 | ✅ Fair |
| 0.051–0.100 | 🟡 Slight bias |
| 0.101–0.200 | 🟠 Moderate bias |
| > 0.200 | 🔴 High bias |

**Current DPD: `{dpd:+.4f}`** → {dpd_verdict(abs(dpd))}
        """, unsafe_allow_html=True)

    st.markdown('<div class="fl-section-title">Group-Level Statistics</div>', unsafe_allow_html=True)
    if group_stats:
        all_rates = [s["positive_rate"] for s in group_stats.values()]
        rows = []
        for g, stats in group_stats.items():
            pr = stats.get("positive_rate", 0)
            rows.append({
                "Group": g, "N (test)": stats.get("n",0),
                "Pred. Positive Rate": f"{pr:.3f}",
                "True Positive Rate":  f"{stats.get('tpr',0):.3f}",
                "Accuracy":            f"{stats.get('accuracy',0):.3f}",
                "Verdict": "Favoured" if pr==max(all_rates) else "Disadvantaged",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Visualizations":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">📈 Bias Visualizations</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Approval Rates", "🎯 Confusion Matrices", "📉 Score Distributions",
        "📈 ROC by Group", "🎯 Calibration", "🌐 Intersectional",
    ])

    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = plot_approval_rates(y_pred, sensitive_test, sensitive_attr)
            st.pyplot(fig); plt.close()
        with col2:
            rates   = [float(y_pred[sensitive_test == g].mean()) for g in np.unique(sensitive_test)]
            max_r   = max(rates); min_r = min(rates); gap = max_r - min_r
            st.markdown(metric_card("Absolute gap", f"{gap:.1%}",
                "🔴 Significant" if gap>0.1 else "🟡 Moderate" if gap>0.05 else "✅ Small",
                "#ffa94d"), unsafe_allow_html=True)

    with tab2:
        fig = plot_confusion_matrices(y_test, y_pred, sensitive_test, sensitive_attr)
        st.pyplot(fig); plt.close()

    with tab3:
        fig = plot_score_distributions(y_pred_proba, sensitive_test, sensitive_attr)
        st.pyplot(fig); plt.close()
        with st.spinner("Computing threshold trade-off…"):
            fig2 = plot_threshold_bias_tradeoff(y_test, y_pred_proba, sensitive_test)
        st.pyplot(fig2); plt.close()

    with tab4:
        fig = plot_roc_by_group(y_test, y_pred_proba, sensitive_test, sensitive_attr)
        st.pyplot(fig); plt.close()

    with tab5:
        fig = plot_calibration(y_test, y_pred_proba, sensitive_test, sensitive_attr)
        st.pyplot(fig); plt.close()

    with tab6:
        try:
            gender_enc = encoders.get("gender"); race_enc = encoders.get("race")
            gender_idx = feature_names.index("gender"); race_idx = feature_names.index("race")
            raw_gender = gender_enc.inverse_transform(X_test[:, gender_idx].astype(int))
            raw_race   = race_enc.inverse_transform(X_test[:, race_idx].astype(int))
            df_test_raw = pd.DataFrame({"gender": raw_gender, "race": raw_race})
            fig = plot_intersectional_heatmap(y_pred, df_test_raw)
            if fig: st.pyplot(fig); plt.close()
        except Exception as e:
            st.info(f"Intersectional view requires gender and race in features. ({e})")

    st.markdown('<div class="fl-section-title">Feature Importance</div>', unsafe_allow_html=True)
    fig = plot_feature_importance(model, feature_names)
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Explainable AI":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">🧠 Explainable AI</h2>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_[0]}).sort_values("Coefficient", key=abs, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(coef_df)*0.42)))
    colors = ["#ff6b6b" if c<0 else "#40c057" for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="#0e0e1f", height=0.65)
    ax.axvline(0, color="#4a4a7a", linewidth=1.2)
    ax.set_title("Logistic Regression Feature Coefficients", fontsize=13, pad=12)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    try:
        import shap
        with st.spinner("Computing SHAP values…"):
            explainer   = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test[:200])
        fig, ax = plt.subplots(figsize=(9,5))
        shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, plot_type="dot", show=False)
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.close()
    except ImportError:
        st.markdown('<div class="fl-alert-info">💡 Install shap for full SHAP analysis.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄  What-If Simulator":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">🔄 What-If Simulator</h2>', unsafe_allow_html=True)
    col_in, col_out = st.columns([1,1], gap="large")
    with col_in:
        age            = st.slider("Age",             18, 75,  35)
        education_num  = st.slider("Education Years",  1, 16,  13)
        hours_per_week = st.slider("Hours / Week",    10, 80,  40)
        capital_gain   = st.select_slider("Capital Gain",  options=[0,2000,5000,10000,15000,50000], value=0)
        capital_loss   = st.select_slider("Capital Loss",  options=[0,1000,2000,4000],              value=0)
        workclass   = st.selectbox("Workclass",      ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov"])
        occupation  = st.selectbox("Occupation",     ["Prof-specialty","Exec-managerial","Craft-repair","Adm-clerical","Sales","Other-service"])
        education   = st.selectbox("Education",      ["Bachelors","Masters","Doctorate","Prof-school","Some-college","HS-grad"])
        marital     = st.selectbox("Marital Status", ["Married-civ-spouse","Never-married","Divorced","Separated","Widowed"])
        gender_sel  = st.radio("Gender", ["Male","Female"], horizontal=True)
        race_sel    = st.selectbox("Race", ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"])
        predict_btn = st.button("🔮  Predict & Compare")

    with col_out:
        if predict_btn:
            user_input = {
                "age":age,"education-num":education_num,"hours-per-week":hours_per_week,
                "capital-gain":capital_gain,"capital-loss":capital_loss,
                "workclass":workclass,"occupation":occupation,"education":education,
                "marital-status":marital,"gender":gender_sel,"race":race_sel,
                "relationship":"Not-in-family","native-country":"United-States",
            }
            res = predict_whatif(model, user_input, feature_names, encoders)
            flip_lbl  = "Female" if gender_sel=="Male" else "Male"
            badge_cls = "pred-badge-pos" if res["prediction"]==1 else "pred-badge-neg"
            badge_txt = "> 50K ✅" if res["prediction"]==1 else "≤ 50K ❌"
            flip_cls  = "pred-badge-pos" if res["flipped_prediction"]==1 else "pred-badge-neg"
            flip_txt  = "> 50K ✅" if res["flipped_prediction"]==1 else "≤ 50K ❌"
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="fl-card" style="text-align:center;padding:1.5rem 1rem;">
                    <div class="fl-card-label">{gender_sel}</div>
                    <div class="{badge_cls}">{badge_txt}</div>
                    {prob_bar(res["probability"], "#6c63ff" if res["prediction"]==1 else "#ff6b6b")}
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="fl-card" style="text-align:center;padding:1.5rem 1rem;">
                    <div class="fl-card-label">{flip_lbl} (flipped)</div>
                    <div class="{flip_cls}">{flip_txt}</div>
                    {prob_bar(res["flipped_probability"], "#6c63ff" if res["flipped_prediction"]==1 else "#ff6b6b")}
                </div>""", unsafe_allow_html=True)
            if res["prediction"] != res["flipped_prediction"]:
                st.markdown(f'<div class="fl-alert-danger">⚠️ <strong>Bias Detected!</strong> Changing only gender {gender_sel} → {flip_lbl} changed the prediction!</div>', unsafe_allow_html=True)
            fig = plot_whatif_comparison(gender_sel, flip_lbl, res["probability"], res["flipped_probability"])
            st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BIAS MITIGATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️  Bias Mitigation":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">🛡️ Bias Mitigation</h2>', unsafe_allow_html=True)
    method = st.selectbox("Method", [
        "Reweighting (Pre-processing) — Recommended",
        "Remove Sensitive Feature (Pre-processing)",
    ], label_visibility="collapsed")

    if st.button("⚙️  Apply Mitigation & Compare"):
        with st.spinner("Training mitigated model…"):
            orig_acc     = evaluate_model(model, X_test, y_test)
            orig_metrics = compute_bias_metrics(y_test, y_pred, sensitive_test)
            orig_score   = compute_bias_score(orig_metrics)
            if method.startswith("Reweight"):
                new_model, new_Xtest, new_ytest = apply_reweighting(
                    X_train, y_train, X_test, y_test, sensitive_test, feature_names, sensitive_attr)
                new_sens = sensitive_test
            else:
                new_model, new_Xtest, new_ytest = apply_remove_sensitive(
                    X_train, y_train, X_test, y_test, feature_names, sensitive_attr)
                new_sens = None
            new_ypred   = new_model.predict(new_Xtest)
            new_acc     = evaluate_model(new_model, new_Xtest, new_ytest)
            new_metrics = compute_bias_metrics(new_ytest, new_ypred, new_sens)
            new_score   = compute_bias_score(new_metrics)

        st.success("✅ Mitigation applied!")
        orig_dpd = abs(orig_metrics.get("demographic_parity_diff",0))
        new_dpd  = abs(new_metrics.get("demographic_parity_diff",0))
        orig_eod = abs(orig_metrics.get("equal_opportunity_diff",0))
        new_eod  = abs(new_metrics.get("equal_opportunity_diff",0))
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Bias Score",  f"{new_score}/100",  delta=f"{new_score-orig_score:+}",  delta_color="inverse")
        with c2: st.metric("|DPD|",       f"{new_dpd:.4f}",    delta=f"{new_dpd-orig_dpd:+.4f}",   delta_color="inverse")
        with c3: st.metric("|EOD|",       f"{new_eod:.4f}",    delta=f"{new_eod-orig_eod:+.4f}",   delta_color="inverse")
        with c4: st.metric("Accuracy",    f"{new_acc:.2%}",    delta=f"{new_acc-orig_acc:+.2%}")
        fig = plot_bias_comparison(orig_metrics, new_metrics, orig_acc, new_acc)
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GEMINI AI ADVISOR  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Gemini AI Advisor":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">✨ Gemini AI Advisor</div>
        <div class="hero-sub">Ask Google Gemini anything about your model's fairness results. Get plain-language explanations, mitigation recommendations, and regulatory guidance — powered by Gemini 1.5 Flash.</div>
        <span class="hero-tag">🤖 Google Gemini 1.5 Flash</span>
        <span class="hero-tag">⚖️ Fairness Advisor</span>
        <span class="hero-tag">📋 Compliance Guidance</span>
    </div>""", unsafe_allow_html=True)

    # ── API Key input ──────────────────────────────────────────────────────────
    col_key, col_info = st.columns([2, 1])
    with col_key:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Paste your key from aistudio.google.com/app/apikey",
            value=st.session_state.get("gemini_api_key", os.environ.get("GEMINI_API_KEY", "")),
            help="Get a free key at aistudio.google.com → Get API Key"
        )
        if api_key:
            st.session_state["gemini_api_key"] = api_key
    with col_info:
        st.markdown("""<div class="fl-alert-info" style="font-size:0.8rem;margin-top:1.6rem;">
        🔑 <strong>Free key:</strong> <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#74c0fc;">aistudio.google.com</a><br/>
        Key stays in your browser session only.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Build live context from current model metrics ──────────────────────────
    acc = evaluate_model(model, X_test, y_test)
    context = build_bias_context(bias_metrics, bias_score_val, acc, sensitive_attr)

    # ── Quick-prompt buttons ───────────────────────────────────────────────────
    st.markdown('<div class="fl-section-title">Quick Questions</div>', unsafe_allow_html=True)
    q_cols = st.columns(3)
    quick_prompts = [
        ("📊 Explain my results",    "Explain what the current bias score and fairness metrics mean in plain language. Is this model suitable for deployment?"),
        ("🛡️ How to fix the bias",  "What are the best mitigation strategies for this level of bias? Compare reweighting, adversarial debiasing, and fairness constraints."),
        ("⚖️ Legal implications",   "What are the legal and regulatory implications of this bias level? Reference EU AI Act, EEOC, and FCRA where relevant."),
        ("🔍 Root cause analysis",  "What are the likely root causes of this bias in the Adult Income Dataset? How does historical inequality propagate into ML models?"),
        ("📋 Write executive summary", "Write a concise 3-paragraph executive summary of these fairness audit results suitable for a non-technical stakeholder."),
        ("🏗️ Deployment checklist", "Give me a fairness checklist for deploying this model responsibly, covering monitoring, documentation, and human oversight."),
    ]
    for i, (label, prompt) in enumerate(quick_prompts):
        with q_cols[i % 3]:
            if st.button(label, key=f"quick_{i}"):
                st.session_state["gemini_input"] = prompt

    # ── Chat interface ─────────────────────────────────────────────────────────
    st.markdown('<div class="fl-section-title">Ask Gemini</div>', unsafe_allow_html=True)

    if "gemini_history" not in st.session_state:
        st.session_state["gemini_history"] = []

    user_question = st.text_area(
        "Your question",
        value=st.session_state.pop("gemini_input", ""),
        placeholder="e.g. 'What does a bias score of 67 mean? Should I be worried?'",
        height=100,
        label_visibility="collapsed"
    )

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_btn = st.button("✨  Ask Gemini", use_container_width=True)
    with col_clear:
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state["gemini_history"] = []
            st.rerun()

    if ask_btn and user_question.strip():
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key above first.")
        else:
            with st.spinner("✨ Gemini is thinking…"):
                response = ask_gemini(
                    question   = user_question,
                    context    = context,
                    api_key    = api_key,
                    history    = st.session_state["gemini_history"],
                )
            st.session_state["gemini_history"].append({
                "user": user_question,
                "ai":   response,
            })

    # ── Render conversation history ────────────────────────────────────────────
    if st.session_state.get("gemini_history"):
        st.markdown('<div class="fl-section-title">Conversation</div>', unsafe_allow_html=True)
        for turn in reversed(st.session_state["gemini_history"]):
            st.markdown(f'<div class="gemini-hdr">You</div><div class="gemini-user">{turn["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gemini-hdr">✨ Gemini</div><div class="gemini-ai">{turn["ai"]}</div>', unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="fl-card" style="text-align:center;padding:2.5rem 1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">✨</div>
            <div style="color:#6868a0;font-size:0.9rem;line-height:1.8;">
                Enter your Gemini API key and ask any question<br>about your model's fairness results.<br>
                <strong style="color:#a78bfa;">Try one of the Quick Questions above to get started.</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Auto-generated report using Gemini ────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="fl-section-title">AI-Generated Fairness Narrative</div>', unsafe_allow_html=True)
    st.caption("Click below to have Gemini write a professional fairness narrative for your audit report.")
    if st.button("📝  Generate AI Narrative", use_container_width=False):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key first.")
        else:
            with st.spinner("✨ Generating narrative…"):
                narrative = ask_gemini(
                    question=(
                        "Write a professional 4-paragraph fairness audit narrative for this model. "
                        "Include: (1) executive summary, (2) key findings with specific numbers, "
                        "(3) risk assessment and regulatory implications, (4) recommended actions. "
                        "Use formal, clear language suitable for a compliance report."
                    ),
                    context=context,
                    api_key=api_key,
                    history=[],
                )
            st.markdown(f'<div class="gemini-ai">{narrative}</div>', unsafe_allow_html=True)
            st.download_button("📥 Download Narrative (.txt)", data=narrative,
                               file_name="gemini_fairness_narrative.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAIRNESS REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄  Fairness Report":
    st.markdown('<h2 style="font-family:\'Space Mono\',monospace;color:#a78bfa;">📄 Fairness Audit Report</h2>', unsafe_allow_html=True)
    acc         = evaluate_model(model, X_test, y_test)
    report_text = generate_report(bias_metrics, bias_score_val, acc, sensitive_attr, len(df))
    dpd = bias_metrics.get("demographic_parity_diff", 0)

    col1, col2 = st.columns([3,1])
    with col1:
        st.text_area("", value=report_text, height=500, label_visibility="collapsed")
    with col2:
        st.download_button("📥  Download (.txt)", data=report_text,
            file_name=f"fairlens_report_{sensitive_attr}.txt", mime="text/plain", use_container_width=True)
        st.markdown(metric_card("Bias Score", f"{bias_score_val}/100",  sc_emoji_str, sc_c), unsafe_allow_html=True)
        st.markdown(metric_card("|DPD|",      f"{abs(dpd):.4f}", "0.05 threshold"),           unsafe_allow_html=True)
        st.markdown(metric_card("Accuracy",   f"{acc:.2%}",      "LR model"),                 unsafe_allow_html=True)
        st.markdown('<div class="fl-alert-info" style="font-size:0.8rem;">💡 Go to ✨ Gemini AI Advisor to generate an AI-enhanced narrative for this report.</div>', unsafe_allow_html=True)
