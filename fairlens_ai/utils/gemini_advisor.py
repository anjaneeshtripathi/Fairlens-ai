"""
utils/gemini_advisor.py
=======================
Google Gemini integration for FairLens AI.

Uses the Gemini REST API directly (no SDK required) so it works
in any Python environment without extra installs.

Model used: gemini-1.5-flash  (free tier, fast, supports long context)
"""

import json
import urllib.request
import urllib.error


# ── System prompt injected into every request ─────────────────────────────────
_SYSTEM_PROMPT = """You are FairLens AI's built-in fairness advisor, powered by Google Gemini.

You are a world-class expert in:
- Algorithmic fairness and AI bias detection
- Machine learning model auditing
- Regulatory compliance (EU AI Act, EEOC, FCRA, GDPR)
- Practical bias mitigation techniques

Your role is to help the user understand their model's fairness audit results,
interpret the metrics, and decide what actions to take.

STYLE GUIDELINES:
- Be clear, direct, and practical
- Use bullet points and numbered lists when listing multiple items
- Cite specific numbers from the context when available
- Always ground recommendations in the actual metrics shown
- When discussing regulations, be accurate but note you are not a lawyer
- Keep responses focused and under 400 words unless asked for something longer
"""


def build_bias_context(
    bias_metrics: dict,
    bias_score: int,
    accuracy: float,
    sensitive_attr: str,
) -> str:
    """
    Build a structured text summary of the current model's fairness
    results to inject as context into every Gemini request.
    """
    dpd = bias_metrics.get("demographic_parity_diff", 0)
    eod = bias_metrics.get("equal_opportunity_diff", 0)
    group_stats = bias_metrics.get("group_stats", {})

    verdict_map = {
        (0, 20):  "Very Fair ✅",
        (21, 40): "Slight Bias 🟡",
        (41, 60): "Moderate Bias 🟠",
        (61, 80): "High Bias 🔴",
        (81, 100):"Severe Bias ⛔",
    }
    verdict = next((v for (lo, hi), v in verdict_map.items() if lo <= bias_score <= hi), "Unknown")

    group_lines = ""
    for g, stats in group_stats.items():
        group_lines += (
            f"  - Group {g}: N={stats.get('n','?')}, "
            f"Positive Rate={stats.get('positive_rate','?'):.3f}, "
            f"TPR={stats.get('tpr','?'):.3f}, "
            f"Accuracy={stats.get('accuracy','?'):.3f}\n"
        )

    context = f"""
=== CURRENT FAIRLENS AI MODEL AUDIT RESULTS ===

Dataset     : Adult Income Dataset (UCI ML Repository)
Records     : 48,842 rows | 14 features
Model       : Logistic Regression (sklearn, class_weight=balanced)
Task        : Binary classification — predict income > $50K/year

Sensitive Attribute Analysed : {sensitive_attr.upper()}

--- FAIRNESS METRICS ---
Bias Score (0–100)                : {bias_score}/100  →  {verdict}
Demographic Parity Difference     : {dpd:+.6f}
Equal Opportunity Difference      : {eod:+.6f}

Bias Score Formula: score = min(100, |DPD|×300 + |EOD|×200)

--- PERFORMANCE ---
Overall Accuracy : {accuracy:.2%}

--- GROUP-LEVEL STATISTICS ---
{group_lines if group_lines else '  (Not available)'}

--- BIAS SCALE REFERENCE ---
0–20   : Very Fair ✅
21–40  : Slight Bias 🟡
41–60  : Moderate Bias 🟠
61–80  : High Bias 🔴
81–100 : Severe Bias ⛔

--- AVAILABLE MITIGATIONS IN THE APP ---
1. Reweighting (pre-processing) — adjusts sample weights per (group × label)
2. Remove Sensitive Feature (pre-processing) — drops protected attribute

=== END OF AUDIT CONTEXT ===
"""
    return context.strip()


def ask_gemini(
    question: str,
    context: str,
    api_key: str,
    history: list = None,
    model: str = "gemini-1.5-flash",
) -> str:
    """
    Send a question to Google Gemini with the fairness context injected.

    Parameters
    ----------
    question : str   — The user's question
    context  : str   — build_bias_context() output
    api_key  : str   — Google AI Studio API key
    history  : list  — List of {"user":..., "ai":...} dicts for multi-turn
    model    : str   — Gemini model name

    Returns
    -------
    str — Gemini's response text, or an error message
    """
    if not api_key or not api_key.strip():
        return "❌ No API key provided. Please enter your Gemini API key."

    history = history or []

    # Build the conversation contents array
    contents = []

    # First turn: inject context + system guidance as the opening user message
    system_user_msg = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Here is the current fairness audit data you should reference:\n\n"
        f"{context}"
    )
    contents.append({
        "role": "user",
        "parts": [{"text": system_user_msg}]
    })
    contents.append({
        "role": "model",
        "parts": [{"text": (
            "Understood. I have the full fairness audit context loaded. "
            "I'm ready to help analyse and advise on this model's bias results. "
            "What would you like to know?"
        )}]
    })

    # Add prior conversation history
    for turn in history[-6:]:   # keep last 6 turns to stay within context
        contents.append({"role": "user",  "parts": [{"text": turn["user"]}]})
        contents.append({"role": "model", "parts": [{"text": turn["ai"]}]})

    # Add current question
    contents.append({"role": "user", "parts": [{"text": question}]})

    # Build request payload
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature":     0.7,
            "maxOutputTokens": 1024,
            "topP":            0.9,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    try:
        req_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data    = req_body,
            method  = "POST",
            headers = {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Extract text from response
        candidates = data.get("candidates", [])
        if not candidates:
            # Check for prompt feedback (blocked)
            feedback = data.get("promptFeedback", {})
            block_reason = feedback.get("blockReason", "Unknown")
            return f"⚠️ Gemini blocked the request: {block_reason}. Try rephrasing your question."

        parts = candidates[0].get("content", {}).get("parts", [])
        text  = " ".join(p.get("text", "") for p in parts).strip()
        return text if text else "⚠️ Gemini returned an empty response. Please try again."

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(body)
            err_msg  = err_json.get("error", {}).get("message", body)
        except Exception:
            err_msg = body[:300]
        if e.code == 400:
            return f"❌ Bad request (400): {err_msg}\n\nCheck your API key and try again."
        if e.code == 403:
            return "❌ API key is invalid or the Gemini API is not enabled for this key. Visit aistudio.google.com to verify."
        if e.code == 429:
            return "⏳ Rate limit reached. Wait a moment and try again (free tier = 15 requests/min)."
        return f"❌ HTTP {e.code}: {err_msg}"

    except urllib.error.URLError as e:
        return f"❌ Network error: {e.reason}. Check your internet connection."

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"
