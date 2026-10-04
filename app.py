import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "sample_credit_applications.csv"

st.set_page_config(
    page_title="CreditWise AI | Intelligent Credit Screening",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --cw-navy: #0b1220;
        --cw-blue: #2563eb;
        --cw-blue-2: #60a5fa;
        --cw-cyan: #06b6d4;
        --cw-green: #10b981;
        --cw-amber: #f59e0b;
        --cw-red: #ef4444;
        --cw-slate: #64748b;
        --cw-border: #e2e8f0;
        --cw-bg: #f8fafc;
    }

    .stApp { background: #f8fafc; }
    .block-container { padding-top: 1.15rem; padding-bottom: 3rem; max-width: 1450px; }

    /* Hide Streamlit chrome that is not useful for the presentation */
    [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
    footer { visibility: hidden; }

    .cw-hero {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.65rem 1.25rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0b1220 0%, #122b68 54%, #2563eb 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
    }
    .cw-hero:after {
        content: "";
        position: absolute;
        right: -80px;
        top: -85px;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(96,165,250,.30), rgba(96,165,250,0) 65%);
    }
    .cw-eyebrow {
        font-size: .74rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 800;
        color: #bfdbfe;
    }
    .cw-hero h1 { margin: .28rem 0 .22rem; font-size: 2.25rem; line-height: 1.05; letter-spacing:-.025em; }
    .cw-hero p { margin: .35rem 0 0; font-size: 1rem; max-width: 920px; color: #dbeafe; }
    .cw-badge-row { display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.75rem; }
    .cw-badge {
        display:inline-flex; align-items:center; gap:.35rem; padding:.38rem .65rem;
        border: 1px solid rgba(255,255,255,.20); border-radius:999px;
        background: rgba(255,255,255,.08); font-size:.78rem; color:#eff6ff;
        backdrop-filter: blur(8px);
    }

    /* High-contrast top navigation */
    .nav-shell {
        margin: .25rem 0 .9rem;
        padding: .5rem .65rem .25rem;
        background: #ffffff;
        border: 1px solid #dbe4f0;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(15,23,42,.05);
    }
    div[data-testid="stRadio"] > label { color: #64748b !important; font-weight: 700; }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: .35rem;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #f8fafc;
        border: 1px solid #dbe4f0;
        border-radius: 12px;
        padding: .58rem .82rem;
        min-height: 40px;
        transition: all .15s ease;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #eff6ff;
        border-color: #93c5fd;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #dbeafe;
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(37,99,235,.12);
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        color: #0f172a !important;
        font-weight: 750;
        font-size: .78rem;
        margin: 0;
    }
    .live-lab {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 18px;
        padding: 1rem 1.1rem .85rem;
        box-shadow: 0 9px 24px rgba(37,99,235,.07);
    }
    .live-lab-head {
        display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;
        margin-bottom:.65rem;
    }
    .live-lab-title { color:#0f172a; font-size:1.18rem; font-weight:800; }
    .live-lab-copy { color:#64748b; font-size:.82rem; }

    /* Dark presentation sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111c33 100%);
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] > div { background: transparent; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    section[data-testid="stSidebar"] .stCaptionContainer p { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ffffff !important; }
    section[data-testid="stSidebar"] hr { border-color: #26344f; }

    .demo-strip {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding:.75rem 1rem; border:1px solid #dbeafe; background:linear-gradient(90deg,#eff6ff,#ffffff);
        border-radius:14px; margin: -.15rem 0 1rem;
    }
    .demo-strip strong { color:#1e3a8a; }
    .demo-strip span { color:#64748b; font-size:.8rem; }


    .section-kicker { font-size: .72rem; font-weight: 800; color:#2563eb; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.12rem; }
    .section-title { font-size: 1.45rem; font-weight: 800; color:#0f172a; margin-bottom:.15rem; }
    .section-subtitle { color:#64748b; margin-bottom: .95rem; }

    .info-card {
        background: #ffffff;
        border: 1px solid var(--cw-border);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.045);
        height: 100%;
    }
    .info-card h4 { margin: 0 0 .35rem; color:#0f172a; }
    .info-card p { margin:0; color:#64748b; line-height:1.55; }
    .info-card .icon { font-size:1.25rem; margin-bottom:.35rem; }

    .kpi {
        background:#fff;
        border:1px solid var(--cw-border);
        border-radius:16px;
        padding: .9rem 1rem;
        box-shadow: 0 7px 18px rgba(15,23,42,.045);
    }
    .kpi-label { color:#64748b; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
    .kpi-value { color:#0f172a; font-weight:800; font-size:1.55rem; margin-top:.16rem; }
    .kpi-note { color:#94a3b8; font-size:.72rem; margin-top:.05rem; }

    .flow-wrap {
        background: #fff;
        border:1px solid var(--cw-border);
        border-radius:18px;
        padding:1rem 1.15rem;
        box-shadow: 0 7px 18px rgba(15,23,42,.045);
    }
    .flow {
        display:grid; grid-template-columns: repeat(5, 1fr); gap:.65rem; align-items:stretch;
    }
    .flow-step {
        border:1px solid #dbeafe; background:linear-gradient(180deg,#eff6ff,#ffffff);
        border-radius:14px; padding:.8rem; min-height:105px;
    }
    .flow-num { font-size:.7rem; font-weight:800; color:#2563eb; }
    .flow-title { font-weight:800; color:#0f172a; font-size:.88rem; margin:.18rem 0; }
    .flow-copy { font-size:.74rem; line-height:1.4; color:#64748b; }

    .status-pill { display:inline-flex; align-items:center; gap:.42rem; padding:.36rem .68rem; border-radius:999px; font-size:.76rem; font-weight:700; }
    .status-good { color:#047857; background:#ecfdf5; border:1px solid #a7f3d0; }
    .status-info { color:#1d4ed8; background:#eff6ff; border:1px solid #bfdbfe; }
    .status-warn { color:#b45309; background:#fffbeb; border:1px solid #fde68a; }
    .dot { width:8px; height:8px; border-radius:50%; background:#10b981; }

    .risk-low-card { border-left: 5px solid #10b981; background:#ecfdf5; }
    .risk-mid-card { border-left: 5px solid #f59e0b; background:#fffbeb; }
    .risk-high-card { border-left: 5px solid #ef4444; background:#fef2f2; }

    .callout {
        border-radius:14px; border:1px solid #dbeafe; background:#eff6ff; padding:.9rem 1rem;
        color:#1e3a8a; line-height:1.5;
    }

    .small-note { color:#64748b; font-size:.82rem; }
    .muted { color:#64748b; }
    .tight { margin-bottom:.25rem !important; }

    @media (max-width: 900px) {
        .flow { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL CONSTANTS
# ============================================================
FEATURES = [
    "age", "annual_income", "employment_years", "loan_amount", "loan_term_months",
    "credit_score", "debt_to_income", "existing_loans", "missed_payments_12m",
    "savings_balance", "employment_type", "housing_status", "dependents"
]
TARGET = "approved"
NUMERIC = [
    "age", "annual_income", "employment_years", "loan_amount", "loan_term_months",
    "credit_score", "debt_to_income", "existing_loans", "missed_payments_12m",
    "savings_balance", "dependents"
]
CATEGORICAL = ["employment_type", "housing_status"]

# ============================================================
# DATA + MODEL
# ============================================================
def _generate_fallback_data(n: int = 2400) -> pd.DataFrame:
    """Generate deterministic demo data if the CSV is unavailable on deployment.

    This makes the public demo self-contained and prevents a missing data file
    from taking down the entire Streamlit application.
    """
    rng = np.random.default_rng(20261004)
    age = rng.integers(21, 66, n)
    annual_income = np.clip(rng.lognormal(mean=np.log(900000), sigma=0.42, size=n), 250000, 6000000)
    employment_years = np.clip((age - 20) * rng.uniform(0.25, 0.75, n), 0, 35)
    loan_amount = np.clip(rng.lognormal(mean=np.log(2200000), sigma=0.65, size=n), 100000, 15000000)
    loan_term_months = rng.choice([24, 36, 48, 60, 72, 84], n, p=[0.07, 0.17, 0.18, 0.32, 0.18, 0.08])
    credit_score = np.clip(rng.normal(700, 75, n), 300, 850).round().astype(int)
    debt_to_income = np.clip(rng.normal(30, 12, n), 2, 75)
    existing_loans = np.clip(rng.poisson(1.3, n), 0, 8)
    missed_payments_12m = np.clip(rng.poisson(0.65, n), 0, 8)
    savings_balance = np.clip(annual_income * rng.uniform(0.03, 0.65, n), 10000, 5000000)
    employment_type = rng.choice(["Salaried", "Self-employed", "Contract"], n, p=[0.64, 0.26, 0.10])
    housing_status = rng.choice(["Owned", "Rented", "Other"], n, p=[0.45, 0.48, 0.07])
    dependents = np.clip(rng.poisson(1.4, n), 0, 6)
    score = (
        -0.2 + 0.006 * (credit_score - 680) - 0.045 * (debt_to_income - 30)
        - 0.42 * missed_payments_12m + 0.12 * employment_years
        - 0.00000008 * loan_amount + 0.00000016 * annual_income
        + 0.00000015 * savings_balance - 0.16 * existing_loans
        - 0.04 * dependents + 0.18 * (employment_type == "Salaried")
        + 0.12 * (housing_status == "Owned")
    )
    prob = 1 / (1 + np.exp(-score))
    approved = rng.binomial(1, prob)
    return pd.DataFrame({
        "age": age, "annual_income": annual_income.round(0).astype(int),
        "employment_years": employment_years.round(1), "loan_amount": loan_amount.round(0).astype(int),
        "loan_term_months": loan_term_months, "credit_score": credit_score,
        "debt_to_income": debt_to_income.round(1), "existing_loans": existing_loans,
        "missed_payments_12m": missed_payments_12m, "savings_balance": savings_balance.round(0).astype(int),
        "employment_type": employment_type, "housing_status": housing_status,
        "dependents": dependents, "approved": approved,
    })


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            data = pd.read_csv(DATA_PATH)
            if len(data) >= 100 and all(c in data.columns for c in FEATURES + [TARGET]):
                return data
        except Exception:
            pass
    return _generate_fallback_data()


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL,
            ),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds, output_dict=True),
    }
    return model, metrics


df = load_data()
model, metrics = train_model(df)

# Cached portfolio predictions help analytics feel "live" but remain deterministic.
@st.cache_data(show_spinner=False)
def portfolio_scored(_df: pd.DataFrame) -> pd.DataFrame:
    out = _df.copy()
    out["pred_probability"] = model.predict_proba(out[FEATURES])[:, 1]
    out["risk_bucket"] = pd.cut(
        out["pred_probability"],
        bins=[-0.001, 0.499999, 0.699999, 1.0],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    )
    return out

portfolio = portfolio_scored(df)

# ============================================================
# HELPERS
# ============================================================
def score_bucket(probability: float) -> Tuple[str, str]:
    if probability >= 0.70:
        return "Low Risk / Pre-Qualify", "low"
    if probability >= 0.50:
        return "Medium Risk / Refer", "medium"
    return "High Risk / Refer", "high"


def rule_flags(a: Dict) -> List[str]:
    flags = []
    if a["credit_score"] < 600:
        flags.append("Credit score below the 600 demonstration threshold")
    if a["debt_to_income"] > 45:
        flags.append("Debt-to-income ratio above 45%")
    if a["missed_payments_12m"] >= 3:
        flags.append("Three or more missed payments in the last 12 months")
    if a["loan_amount"] > a["annual_income"] * 0.45:
        flags.append("Requested loan is high relative to annual income")
    if a["savings_balance"] < a["loan_amount"] * 0.10:
        flags.append("Low liquid savings buffer relative to requested loan")
    return flags


def recommendation(probability: float, flags: List[str]) -> str:
    if probability >= 0.70 and not flags:
        return "Pre-qualify for the next underwriting stage. Verify documents and affordability before any final approval."
    if probability >= 0.50:
        return "Refer for manual review. Confirm income, affordability, repayment history, and supporting documents."
    return "Do not auto-approve. Route to manual review or decline workflow according to the lender's formal credit policy."


def local_explanation(a: Dict, probability: float, flags: List[str]) -> str:
    pct = probability * 100
    positives, cautions = [], []
    if a["credit_score"] >= 750:
        positives.append("strong credit score")
    elif a["credit_score"] >= 700:
        positives.append("healthy credit score")
    if a["debt_to_income"] <= 30:
        positives.append("low debt-to-income ratio")
    elif a["debt_to_income"] > 45:
        cautions.append("high debt-to-income ratio")
    if a["missed_payments_12m"] == 0:
        positives.append("clean recent repayment history")
    elif a["missed_payments_12m"] >= 3:
        cautions.append("multiple missed payments")
    if a["employment_years"] >= 5:
        positives.append("stable employment history")
    if a["loan_amount"] <= a["annual_income"] * 0.25:
        positives.append("moderate loan-to-income request")
    elif a["loan_amount"] > a["annual_income"] * 0.45:
        cautions.append("large loan relative to income")
    if not positives:
        positives.append("no major positive driver detected")
    if not cautions:
        cautions.append("no major rule-based caution flag detected")
    bucket, _ = score_bucket(probability)
    return (
        f"Model-estimated approval probability: {pct:.1f}%. Classification: {bucket}. "
        f"Positive signals: {', '.join(positives)}. Caution signals: {', '.join(cautions)}. "
        "This is a screening aid, not a final lending decision."
    )


def model_contributions(applicant: Dict) -> List[Tuple[str, float]]:
    pipe = model
    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]
    x_df = pd.DataFrame([applicant])[FEATURES]
    transformed = pre.transform(x_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = transformed[0]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        return []
    coefs = clf.coef_[0]
    contrib = transformed * coefs
    pairs = sorted(zip(names, contrib), key=lambda z: abs(z[1]), reverse=True)
    readable = []
    for name, val in pairs[:8]:
        cleaned = name.replace("num__", "").replace("cat__", "")
        readable.append((cleaned, float(val)))
    return readable


def get_gemini_api_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = None
    return key or os.getenv("GEMINI_API_KEY")


def generate_gemini_explanation(a: Dict, probability: float, flags: List[str], recommendation_text: str) -> Tuple[str, str]:
    api_key = get_gemini_api_key()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    if not api_key:
        return local_explanation(a, probability, flags), "Local fallback"
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are CreditWise AI, an explainability layer for a credit-risk pre-screening prototype.
Never claim to make a final lending decision. Do not invent facts. Use only the supplied data.
Explain the model result in plain business language suitable for an MBA viva.
Do not use sensitive/protected attributes. Do not tell the user to manipulate the application.

Applicant data:
{json.dumps(a, indent=2)}

Model estimated approval probability: {probability*100:.1f}%
Rule flags: {flags if flags else 'None'}
Recommended next action: {recommendation_text}

Give: 1) one-sentence conclusion; 2) 3 concise reasons tied to the input data;
3) one limitation/caveat; 4) one safe next step. Keep it under 170 words.
"""
        resp = client.models.generate_content(model=model_name, contents=prompt)
        text = getattr(resp, "text", None) or str(resp)
        return text.strip(), f"Gemini ({model_name})"
    except Exception as exc:
        return local_explanation(a, probability, flags) + f"\n\nGemini note: unavailable ({type(exc).__name__}); local explanation used instead.", "Local fallback after Gemini error"


def copilot_response(question: str, context_text: str = "") -> Tuple[str, str]:
    api_key = get_gemini_api_key()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    system = """
You are CreditWise AI Copilot. You explain a classroom credit-risk pre-screening prototype.
Scope: explain screening logic, risk drivers, model metrics, what-if analysis, and app usage.
Out of scope: binding financial advice, legal advice, final credit decisions, or personal loan shopping.
Do not invent policies. If the question is outside scope, say so briefly and redirect.
"""
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"{system}\n\nCurrent app context:\n{context_text}\n\nUser question: {question}"
            resp = client.models.generate_content(model=model_name, contents=prompt)
            return (getattr(resp, "text", None) or str(resp)).strip(), f"Gemini ({model_name})"
        except Exception:
            pass
    q = question.lower()
    if "score" in q or "probability" in q:
        return ("CreditWise combines the applicant inputs in a logistic-regression model and reports an estimated approval probability. The app then applies transparent rule checks such as high DTI, low credit score, or multiple recent missed payments. The score supports screening; it does not replace formal underwriting.", "Local copilot")
    if "gemini" in q or "api" in q:
        return ("Gemini is optional in this prototype. With a GEMINI_API_KEY configured, it generates plain-language explanations and Copilot answers. Without the key, the app remains functional through its local model and rule-based explanation.", "Local copilot")
    if "what-if" in q or "what if" in q:
        return ("Use the What-If Simulator to change credit score, DTI, loan amount, income, or missed payments and compare the resulting model probability with the baseline applicant.", "Local copilot")
    return ("I can explain the screening score, risk flags, model metrics, data fields, and what-if simulator. For questions outside those areas, use the app only as an educational prototype and consult a qualified professional.", "Local copilot")


def money_in_lakh(value: float) -> str:
    return f"₹{value / 100000:.1f}L"


def risk_card_html(title: str, probability: str, text: str, css_class: str) -> str:
    return f"<div class='info-card {css_class}'><div class='small-note'>{title}</div><h3 style='margin:.25rem 0;color:#0f172a'>{probability}</h3><div class='muted'>{text}</div></div>"


# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
    <div class='cw-hero'>
      <div class='cw-eyebrow'>MBA Finance + Big Data Analytics • End-Term Prototype</div>
      <h1>CreditWise AI</h1>
      <p>Intelligent credit-risk pre-screening with explainable machine learning, transparent risk guardrails, what-if analysis and an AI Copilot.</p>
      <div class='cw-badge-row'>
        <span class='cw-badge'>● Model live</span>
        <span class='cw-badge'>✦ 2,400 synthetic applications</span>
        <span class='cw-badge'>◇ Logistic Regression + Rule Layer</span>
        <span class='cw-badge'>AI explanation: {'Enabled' if get_gemini_api_key() else 'Local fallback'}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## CreditWise AI")
    st.caption("Decision-support prototype for credit screening")
    st.markdown("---")
    st.markdown("**Live system status**")
    st.markdown("<span class='status-pill status-good'><span class='dot'></span> Model loaded</span>", unsafe_allow_html=True)
    if get_gemini_api_key():
        st.markdown("<span class='status-pill status-good' style='margin-top:.35rem'><span class='dot'></span> Gemini connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill status-info' style='margin-top:.35rem'>○ Gemini optional</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Project navigation**")
    st.caption("Use the navigation bar below the header. The What-If Lab is interactive with live sliders.")

    st.markdown("---")
    st.markdown("**Key model indicators**")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("Holdout accuracy", f"{metrics['accuracy']*100:.1f}%")
    st.metric("Portfolio approval rate", f"{df[TARGET].mean()*100:.1f}%")

    st.markdown("---")
    st.markdown("**Safety note**")
    st.caption("Synthetic classroom data only. This application supports screening education and should not be used as a production lending or financial-advice system.")

# ============================================================
# QUICK DEMO STRIP
# ============================================================
st.markdown(
    """<div class='demo-strip'><div><strong>Recommended 3-minute demo:</strong> Screen → Explain → What-If → Portfolio → Copilot</div><span>Best viewed at 100% browser zoom</span></div>""",
    unsafe_allow_html=True,
)

# ============================================================
# HIGH-VISIBILITY NAVIGATION
# ============================================================
NAV_OPTIONS = [
    "Overview", "Applicant Screening", "What-If Lab", "Portfolio Intelligence", "AI Copilot", "Project Guide"
]
st.markdown("<div class='nav-shell'>", unsafe_allow_html=True)
DEMO_VALUES = {
    "Demo — Strong profile": {
        "age": 34, "annual_income": 1500000, "employment_years": 8, "loan_amount": 2500000,
        "loan_term_months": 60, "credit_score": 790, "debt_to_income": 22.0, "existing_loans": 1,
        "missed_payments_12m": 0, "savings_balance": 700000, "employment_type": "Salaried", "housing_status": "Owned", "dependents": 1
    },
    "Demo — Borderline profile": {
        "age": 31, "annual_income": 1000000, "employment_years": 4, "loan_amount": 2500000,
        "loan_term_months": 60, "credit_score": 670, "debt_to_income": 34.0, "existing_loans": 1,
        "missed_payments_12m": 0, "savings_balance": 250000, "employment_type": "Salaried", "housing_status": "Rented", "dependents": 2
    },
    "Demo — High-risk profile": {
        "age": 28, "annual_income": 550000, "employment_years": 2, "loan_amount": 3500000,
        "loan_term_months": 48, "credit_score": 545, "debt_to_income": 58.0, "existing_loans": 4,
        "missed_payments_12m": 4, "savings_balance": 50000, "employment_type": "Self-employed", "housing_status": "Rented", "dependents": 3
    },
}

page = st.radio(
    "Navigate through the CreditWise workflow",
    NAV_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    index=0,
)
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown("<div class='section-kicker'>Executive view</div><div class='section-title'>A complete AI-enabled credit screening workflow</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Start here for the business context, model health and end-to-end architecture.</div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Applications in demo</div><div class='kpi-value'>{len(df):,}</div><div class='kpi-note'>Synthetic records</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Model ROC-AUC</div><div class='kpi-value'>{metrics['roc_auc']:.3f}</div><div class='kpi-note'>Holdout validation</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Observed approval rate</div><div class='kpi-value'>{df[TARGET].mean()*100:.1f}%</div><div class='kpi-note'>Historical synthetic label</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Average loan request</div><div class='kpi-value'>{money_in_lakh(df['loan_amount'].mean())}</div><div class='kpi-note'>Across demo portfolio</div></div>", unsafe_allow_html=True)

    st.write("")
    f1, f2, f3, f4, f5 = st.columns(5)
    flow_items = [
        ("01", "Applicant input", "Income, credit score, DTI, repayment history and loan request."),
        ("02", "ML score", "Logistic Regression estimates approval probability on the prototype data."),
        ("03", "Rule guardrails", "Transparent checks catch high DTI, missed payments and weak buffers."),
        ("04", "Explain + compare", "Plain-language explanation plus local feature contributions and what-if analysis."),
        ("05", "Human workflow", "Pre-qualify, refer or escalate — never a final lending decision."),
    ]
    for col, (num, title, copy) in zip([f1, f2, f3, f4, f5], flow_items):
        with col:
            st.markdown(f"<div class='info-card'><div class='flow-num'>{num}</div><h4>{title}</h4><p>{copy}</p></div>", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("<div class='section-kicker'>Why this is strong for the assignment</div><div class='section-title' style='font-size:1.2rem'>Finance problem + analytics + AI</div>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        cards = [
            ("💼", "Business value", "Speeds up first-level loan triage while making the logic easier to explain."),
            ("📊", "BDA value", "Combines structured data, predictive modelling, validation metrics and portfolio analytics."),
            ("✨", "AI value", "Uses Gemini as an explanation/copilot layer rather than allowing a generative model to make the credit decision."),
        ]
        for col, (icon, title, copy) in zip([a, b, c], cards):
            with col:
                st.markdown(f"<div class='info-card'><div class='icon'>{icon}</div><h4>{title}</h4><p>{copy}</p></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-kicker'>Model health</div><div class='section-title' style='font-size:1.2rem'>Holdout validation snapshot</div>", unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['roc_auc'] * 100,
            number={'suffix': '%', 'font': {'size': 32}},
            title={'text': 'ROC-AUC on holdout', 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#2563eb'},
                'steps': [
                    {'range': [0, 60], 'color': '#fee2e2'},
                    {'range': [60, 70], 'color': '#fef3c7'},
                    {'range': [70, 85], 'color': '#dbeafe'},
                    {'range': [85, 100], 'color': '#dcfce7'},
                ],
            }
        ))
        gauge.update_layout(height=225, margin=dict(l=20, r=20, t=35, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<div class='callout'><b>Interpretation:</b> the prototype separates higher- and lower-risk cases better than chance, but the model is trained on synthetic data and should be treated as a classroom demonstration.</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class='live-lab'>
      <div class='live-lab-head'>
        <div>
          <div class='section-kicker'>Live demo control room</div>
          <div class='live-lab-title'>Try the model without leaving the home screen</div>
          <div class='live-lab-copy'>Move the sliders and the approval probability updates instantly. This is the easiest feature to demonstrate in your viva.</div>
        </div>
        <span class='status-pill status-info'>● Interactive</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        ov_credit = st.slider("Credit score", 300, 900, 790, 1, key="overview_credit")
    with ov2:
        ov_dti = st.slider("Debt-to-income (%)", 0.0, 80.0, 22.0, 0.5, key="overview_dti")
    with ov3:
        ov_income = st.slider("Annual income (₹ lakh)", 1.5, 100.0, 15.0, 0.5, key="overview_income_lakh")
    with ov4:
        ov_loan = st.slider("Loan request (₹ lakh)", 0.5, 300.0, 25.0, 0.5, key="overview_loan_lakh")

    ov_base = {
        "age": 34, "annual_income": 1500000, "employment_years": 8, "loan_amount": 2500000,
        "loan_term_months": 60, "credit_score": ov_credit, "debt_to_income": ov_dti, "existing_loans": 1,
        "missed_payments_12m": 0, "savings_balance": 700000, "employment_type": "Salaried",
        "housing_status": "Owned", "dependents": 1,
    }
    ov_base_prob = float(model.predict_proba(pd.DataFrame([{**ov_base, "credit_score": 790, "debt_to_income": 22.0, "annual_income": 1500000, "loan_amount": 2500000}])[FEATURES])[0,1])
    ov_live_prob = float(model.predict_proba(pd.DataFrame([ov_base])[FEATURES])[0,1])
    ov_delta = (ov_live_prob - ov_base_prob) * 100
    ov_flags = rule_flags(ov_base)
    ov_bucket, _ov_css = score_bucket(ov_live_prob)
    om1, om2, om3, om4 = st.columns(4)
    with om1: st.metric("Baseline", f"{ov_base_prob*100:.1f}%")
    with om2: st.metric("Live probability", f"{ov_live_prob*100:.1f}%")
    with om3: st.metric("Change", f"{ov_delta:+.1f} pp", delta=f"{ov_delta:+.1f} pp")
    with om4: st.metric("Risk bucket", ov_bucket.split(" /")[0])
    ov_left, ov_right = st.columns([1.05, .95])
    with ov_left:
        mini = go.Figure(go.Indicator(
            mode="gauge+number", value=ov_live_prob * 100, number={"suffix":"%", "font":{"size":30}},
            title={"text":"Live approval probability", "font":{"size":13}},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":"#2563eb"},
                   "steps":[{"range":[0,50],"color":"#fee2e2"},{"range":[50,70],"color":"#fef3c7"},{"range":[70,100],"color":"#dcfce7"}]},
        ))
        mini.update_layout(height=250, margin=dict(l=20,r=20,t=35,b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(mini, use_container_width=True, config={"displayModeBar":False})
    with ov_right:
        st.markdown("**Live guardrails**")
        if ov_flags:
            for flag in ov_flags[:4]:
                st.warning(flag)
        else:
            st.success("No demonstration rule-based red flags detected.")
        st.markdown("**How to explain this:**")
        st.write("The sliders modify key financial inputs, the logistic-regression model recalculates probability, and the rule layer independently checks for obvious risk conditions.")

    st.write("")
    st.markdown("<div class='section-kicker'>Risk framework</div><div class='section-title' style='font-size:1.2rem'>How to read the screening result</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(risk_card_html("LOW RISK", "≥ 70%", "Model probability supports moving to the next underwriting step, subject to document and affordability checks.", "risk-low-card"), unsafe_allow_html=True)
    with r2:
        st.markdown(risk_card_html("MEDIUM RISK", "50%–69.9%", "The case is better suited to manual review where a human can validate affordability and supporting evidence.", "risk-mid-card"), unsafe_allow_html=True)
    with r3:
        st.markdown(risk_card_html("HIGH RISK", "< 50%", "Avoid auto-approval in this prototype and route the case through the formal review workflow.", "risk-high-card"), unsafe_allow_html=True)

    st.write("")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Suggested 3-minute demo order**")
        st.write("1. Screen a strong applicant → 2. Show explanations → 3. Change credit score/DTI in What-If → 4. Show portfolio metrics → 5. Ask the Copilot a finance question.")
    with d2:
        st.download_button(
            "Download sample dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="creditwise_sample_credit_applications.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ============================================================
# SCREEN APPLICANT
# ============================================================
elif page == "Applicant Screening":
    st.markdown("<div class='section-kicker'>Decision workspace</div><div class='section-title'>Applicant Screening</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Enter a synthetic profile to see the model score, rule-based guardrails, recommended workflow and explanation.</div>", unsafe_allow_html=True)

    demo_names = ["Custom input", *DEMO_VALUES.keys()]
    demo = st.selectbox("Quick demo profile", demo_names, help="Use a pre-built profile for a fast classroom demonstration.")
    v = DEMO_VALUES.get(demo, {})

    with st.form("screen_form"):
        st.markdown("**Applicant profile**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 75, int(v.get("age", 32)), 1)
            annual_income = st.number_input("Annual income (INR)", 150000, 10000000, int(v.get("annual_income", 1200000)), 50000)
            employment_years = st.number_input("Employment history (years)", 0.0, 40.0, float(v.get("employment_years", 5)), 0.5)
            employment_type = st.selectbox("Employment type", ["Salaried", "Self-employed", "Contract"], index=["Salaried", "Self-employed", "Contract"].index(v.get("employment_type", "Salaried")))
        with c2:
            loan_amount = st.number_input("Requested loan (INR)", 50000, 30000000, int(v.get("loan_amount", 3000000)), 50000)
            loan_term_months = st.slider("Loan term (months)", 12, 120, int(v.get("loan_term_months", 60)), 6)
            credit_score = st.slider("Credit score", 300, 900, int(v.get("credit_score", 720)), 1)
            debt_to_income = st.slider("Debt-to-income (%)", 0.0, 80.0, float(v.get("debt_to_income", 28.0)), 0.5)
        with c3:
            existing_loans = st.number_input("Existing loans", 0, 10, int(v.get("existing_loans", 1)), 1)
            missed_payments = st.number_input("Missed payments (last 12 months)", 0, 12, int(v.get("missed_payments_12m", 0)), 1)
            savings_balance = st.number_input("Savings balance (INR)", 0, 10000000, int(v.get("savings_balance", 500000)), 25000)
            housing_status = st.selectbox("Housing status", ["Owned", "Rented", "Other"], index=["Owned", "Rented", "Other"].index(v.get("housing_status", "Owned")))
            dependents = st.number_input("Dependents", 0, 10, int(v.get("dependents", 1)), 1)

        submitted = st.form_submit_button("▶ Run AI pre-screen", type="primary", use_container_width=True)

    if submitted:
        applicant = {
            "age": age, "annual_income": annual_income, "employment_years": employment_years,
            "loan_amount": loan_amount, "loan_term_months": loan_term_months, "credit_score": credit_score,
            "debt_to_income": debt_to_income, "existing_loans": existing_loans,
            "missed_payments_12m": missed_payments, "savings_balance": savings_balance,
            "employment_type": employment_type, "housing_status": housing_status, "dependents": dependents,
        }
        row = pd.DataFrame([applicant])
        probability = float(model.predict_proba(row[FEATURES])[0, 1])
        bucket, css_class = score_bucket(probability)
        flags = rule_flags(applicant)
        rec = recommendation(probability, flags)
        explanation, source = generate_gemini_explanation(applicant, probability, flags, rec)
        contributions = model_contributions(applicant)
        st.session_state.update({
            "last_applicant": applicant,
            "last_probability": probability,
            "last_explanation": explanation,
            "last_source": source,
            "last_flags": flags,
            "last_recommendation": rec,
            "last_contributions": contributions,
        })

    if "last_probability" in st.session_state:
        probability = st.session_state["last_probability"]
        bucket, css_class = score_bucket(probability)
        flags = st.session_state.get("last_flags", [])
        rec = st.session_state.get("last_recommendation", "")

        st.markdown("<div class='section-kicker' style='margin-top:1.1rem'>Screening output</div><div class='section-title' style='font-size:1.22rem'>Decision-support summary</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Approval probability", f"{probability*100:.1f}%")
        with m2: st.metric("Risk classification", bucket.split(" /")[0])
        with m3: st.metric("Rule flags", str(len(flags)))
        with m4: st.metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}")

        left, mid, right = st.columns([1.05, 1.2, 1.05])
        with left:
            st.markdown("**Risk gauge**")
            gauge_color = "#10b981" if css_class == "low" else "#f59e0b" if css_class == "medium" else "#ef4444"
            risk_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': '%', 'font': {'size': 32}},
                gauge={'axis': {'range':[0,100]}, 'bar': {'color': gauge_color}, 'steps':[{'range':[0,50],'color':'#fee2e2'}, {'range':[50,70],'color':'#fef3c7'}, {'range':[70,100],'color':'#dcfce7'}]},
            ))
            risk_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=10,b=5), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(risk_gauge, use_container_width=True, config={'displayModeBar': False})
            st.caption("Probability shown is the model's estimated approval likelihood on this prototype's synthetic data.")
        with mid:
            st.markdown("**Recommended workflow action**")
            st.info(rec)
            st.markdown("**Why this case landed here**")
            st.write(st.session_state.get("last_explanation", ""))
            st.caption(f"Explanation source: {st.session_state.get('last_source', 'Local')}")
        with right:
            st.markdown("**Guardrail checks**")
            if flags:
                for flag in flags:
                    st.warning(flag)
            else:
                st.success("No demonstration rule-based red flags detected.")

        contributions = st.session_state.get("last_contributions", [])
        if contributions:
            chart_df = pd.DataFrame(contributions, columns=["Feature", "Contribution"])
            chart_df["Direction"] = np.where(chart_df["Contribution"] >= 0, "Supports approval", "Pushes risk higher")
            chart_df = chart_df.sort_values("Contribution")
            fig = px.bar(
                chart_df,
                x="Contribution", y="Feature", color="Direction", orientation="h",
                color_discrete_map={"Supports approval":"#10b981", "Pushes risk higher":"#ef4444"},
                title="Approximate local feature contribution",
                labels={"Contribution":"Contribution to model score"},
            )
            fig.update_layout(height=390, margin=dict(l=20,r=20,t=55,b=20), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("How should I explain this chart in a viva?"):
                st.write("Bars to the right represent inputs that push the logistic-regression score toward approval; bars to the left push it toward higher risk. This is an approximate local explanation of the fitted model, not a causal statement.")

# ============================================================
# WHAT-IF
# ============================================================
elif page == "What-If Lab":
    st.markdown("<div class='section-kicker'>Interactive sensitivity lab</div><div class='section-title'>What-If Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Change financial assumptions and immediately see how the model probability and transparent guardrails respond.</div>", unsafe_allow_html=True)
    st.info("🎛️ **This page is fully interactive:** move the blue sliders below and the probability, risk bucket and guardrails update automatically.")

    base = st.session_state.get("last_applicant")
    if base is None:
        base = DEMO_VALUES.get("Demo — Strong profile", {
            "age": 34, "annual_income": 1500000, "employment_years": 8, "loan_amount": 2500000,
            "loan_term_months": 60, "credit_score": 790, "debt_to_income": 22.0, "existing_loans": 1,
            "missed_payments_12m": 0, "savings_balance": 700000, "employment_type": "Salaried", "housing_status": "Owned", "dependents": 1
        })

    st.markdown("**Adjust the six variables below**")
    w1, w2, w3 = st.columns(3)
    with w1:
        wi_credit = st.slider("Credit score", 300, 900, int(base["credit_score"]), 1)
        wi_dti = st.slider("Debt-to-income (%)", 0.0, 80.0, float(base["debt_to_income"]), 0.5)
    with w2:
        wi_income = st.number_input("Annual income (INR)", 150000, 10000000, int(base["annual_income"]), 50000)
        wi_loan = st.number_input("Loan amount (INR)", 50000, 30000000, int(base["loan_amount"]), 50000)
    with w3:
        wi_missed = st.number_input("Missed payments (12m)", 0, 12, int(base["missed_payments_12m"]), 1)
        wi_savings = st.number_input("Savings balance (INR)", 0, 10000000, int(base["savings_balance"]), 25000)

    what_if = dict(base)
    what_if.update({"credit_score": wi_credit, "debt_to_income": wi_dti, "annual_income": wi_income, "loan_amount": wi_loan, "missed_payments_12m": wi_missed, "savings_balance": wi_savings})
    base_prob = float(model.predict_proba(pd.DataFrame([base])[FEATURES])[0, 1])
    what_if_prob = float(model.predict_proba(pd.DataFrame([what_if])[FEATURES])[0, 1])
    delta = (what_if_prob - base_prob) * 100

    a, b, c, d = st.columns(4)
    with a: st.metric("Baseline probability", f"{base_prob*100:.1f}%")
    with b: st.metric("What-if probability", f"{what_if_prob*100:.1f}%")
    with c: st.metric("Change", f"{delta:+.1f} pp", delta=f"{delta:+.1f} pp")
    with d: st.metric("Risk bucket", score_bucket(what_if_prob)[0].split(" /")[0])

    compare = pd.DataFrame({"Scenario": ["Baseline", "What-if"], "Approval probability": [base_prob, what_if_prob]})
    fig = px.bar(compare, x="Scenario", y="Approval probability", range_y=[0, 1], text_auto=".1%", title="Baseline vs. What-If approval probability")
    fig.update_traces(marker_color=["#94a3b8", "#2563eb"])
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=55,b=20), yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transparent rule layer**")
        what_flags = rule_flags(what_if)
        if what_flags:
            for flag in what_flags:
                st.warning(flag)
        else:
            st.success("No demonstration rule-based red flags detected.")
    with col2:
        st.markdown("**Viva interpretation**")
        direction = "improved" if delta > 0.5 else "weakened" if delta < -0.5 else "changed only slightly"
        st.markdown(f"<div class='callout'>The scenario <b>{direction}</b> the model's estimated approval probability by <b>{abs(delta):.1f} percentage points</b>. Use this to explain model sensitivity, not causality.</div>", unsafe_allow_html=True)

    with st.expander("Why this feature is useful"):
        st.write("The What-If Lab makes the project more than a static classifier: an examiner can change a financial input and immediately see the downstream effect on probability, risk bucket and guardrails.")

# ============================================================
# ANALYTICS
# ============================================================
elif page == "Portfolio Intelligence":
    st.markdown("<div class='section-kicker'>Portfolio intelligence</div><div class='section-title'>Portfolio Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Move from one applicant to the overall synthetic portfolio: distribution, affordability pressure, model performance and predicted risk mix.</div>", unsafe_allow_html=True)

    low = (portfolio["risk_bucket"] == "Low Risk").sum()
    med = (portfolio["risk_bucket"] == "Medium Risk").sum()
    high = (portfolio["risk_bucket"] == "High Risk").sum()
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1: st.metric("Sample applications", f"{len(df):,}")
    with p2: st.metric("Observed approval", f"{df[TARGET].mean()*100:.1f}%")
    with p3: st.metric("Avg. income", money_in_lakh(df["annual_income"].mean()))
    with p4: st.metric("Avg. loan request", money_in_lakh(df["loan_amount"].mean()))
    with p5: st.metric("Avg. DTI", f"{df['debt_to_income'].mean():.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(df, x="credit_score", color=TARGET, nbins=20, barmode="overlay", title="Credit score distribution by observed outcome", labels={TARGET:"Observed approved"}, color_discrete_sequence=["#94a3b8", "#2563eb"])
        fig1.update_layout(height=370, margin=dict(l=20,r=20,t=55,b=20), legend_title_text="")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.scatter(portfolio.sample(min(600, len(portfolio)), random_state=42), x="debt_to_income", y="loan_amount", color="risk_bucket", size="annual_income", hover_data=["credit_score", "missed_payments_12m", "pred_probability"], title="Affordability pressure: DTI vs. requested loan", color_discrete_map={"Low Risk":"#10b981", "Medium Risk":"#f59e0b", "High Risk":"#ef4444"})
        fig2.update_layout(height=370, margin=dict(l=20,r=20,t=55,b=20), legend_title_text="Predicted risk")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns([1, 1.1])
    with c3:
        risk_mix = pd.DataFrame({"Predicted bucket":["Low Risk","Medium Risk","High Risk"], "Applications":[low,med,high]})
        fig3 = px.pie(risk_mix, names="Predicted bucket", values="Applications", hole=.55, title="Model-estimated portfolio risk mix", color="Predicted bucket", color_discrete_map={"Low Risk":"#10b981","Medium Risk":"#f59e0b","High Risk":"#ef4444"})
        fig3.update_layout(height=360, margin=dict(l=20,r=20,t=55,b=20), showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        st.markdown("**Validation metrics**")
        cm = metrics["confusion"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True, hide_index=False)
        q1, q2 = st.columns(2)
        with q1: st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with q2: st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        st.caption("Holdout validation uses a 25% stratified split. Accuracy and ROC-AUC describe this synthetic prototype only.")

    st.markdown("**Selected portfolio characteristics**")
    summary = pd.DataFrame({
        "Metric": ["Average credit score", "Average DTI", "Average savings", "Average missed payments", "Average employment history"],
        "Value": [f"{df['credit_score'].mean():.0f}", f"{df['debt_to_income'].mean():.1f}%", money_in_lakh(df['savings_balance'].mean()), f"{df['missed_payments_12m'].mean():.2f}", f"{df['employment_years'].mean():.1f} years"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ============================================================
# COPILOT
# ============================================================
elif page == "AI Copilot":
    st.markdown("<div class='section-kicker'>Generative AI layer</div><div class='section-title'>AI Copilot</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Ask questions about the screening logic, model interpretation, portfolio metrics, What-If Lab or project architecture.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    prompt_cards = [
        ("Explain the score", "Why does DTI matter in the screening score?"),
        ("Explain the model", "What is logistic regression doing here?"),
        ("Explain the project", "Why is Gemini not allowed to make the credit decision?"),
    ]
    for col, (title, prompt) in zip([c1,c2,c3], prompt_cards):
        with col:
            st.markdown(f"<div class='info-card'><div class='small-note'>{title}</div><div style='font-weight:700;color:#0f172a;margin:.25rem 0 .45rem'>{prompt}</div></div>", unsafe_allow_html=True)

    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = []
    context = {
        "last_probability": st.session_state.get("last_probability"),
        "last_flags": st.session_state.get("last_flags"),
        "model_accuracy": round(metrics["accuracy"], 3),
        "model_roc_auc": round(metrics["roc_auc"], 3),
    }
    for role, msg in st.session_state["copilot_history"]:
        with st.chat_message(role):
            st.write(msg)

    question = st.chat_input("Ask a project question…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        answer, source = copilot_response(question, json.dumps(context))
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"Response source: {source}")
        st.session_state["copilot_history"].append(("user", question))
        st.session_state["copilot_history"].append(("assistant", answer))

    with st.expander("Recommended viva questions"):
        st.write("• Why did you choose logistic regression?  • What does ROC-AUC mean?  • Why use a separate rule layer?  • Why is Gemini optional?  • What are the limitations of synthetic data?  • How would you productionise this system?")

# ============================================================
# PROJECT GUIDE
# ============================================================
elif page == "Project Guide":
    st.markdown("<div class='section-kicker'>Documentation</div><div class='section-title'>Project Guide</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Everything you need to understand the prototype, explain it in a viva and discuss future improvements.</div>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("### 1. Business problem")
        st.write("Lenders need a fast, consistent first-level way to prioritise applications for the next stage of underwriting. CreditWise AI demonstrates how structured applicant data can be turned into an interpretable screening score.")
        st.markdown("### 2. Why hybrid AI?")
        st.write("The predictive model estimates a probability; deterministic rules act as transparent guardrails; Gemini turns the result into a human-readable explanation. This separation improves auditability for a classroom prototype.")
        st.markdown("### 3. Data")
        st.write("The application uses 2,400 synthetic records with features covering income, employment, loan request, credit behaviour, debt burden, savings and household context. No real customer records are used.")
    with a2:
        st.markdown("### 4. Technology")
        st.write("Streamlit • Python • pandas • scikit-learn • Plotly • optional Google Gemini API")
        st.markdown("### 5. Model")
        st.write("Logistic Regression is used because the output is probabilistic and the coefficients can be translated into a local contribution chart. A 25% stratified holdout set is used for validation.")
        st.markdown("### 6. Important limitation")
        st.write("Performance, fairness and business thresholds on synthetic classroom data cannot be generalised to real lending. A production system would need verified data, formal policy rules, bias/fairness testing, drift monitoring, privacy controls, security, audit trails and human oversight.")

    with st.expander("Data dictionary"):
        dd = pd.DataFrame({
            "Field": FEATURES,
            "Meaning": [
                "Applicant age", "Annual income in INR", "Employment history in years", "Requested loan amount in INR", "Requested term in months",
                "Credit score", "Debt-to-income ratio", "Number of existing loans", "Missed payments in previous 12 months", "Savings balance in INR",
                "Employment category", "Housing category", "Number of dependents"
            ]
        })
        st.dataframe(dd, use_container_width=True, hide_index=True)

    with st.expander("Production roadmap"):
        roadmap = pd.DataFrame({
            "Area": ["Data", "Risk model", "Explainability", "Governance", "Deployment", "Monitoring"],
            "Next step": [
                "Connect verified bureau and bank data with consent and data-quality checks",
                "Compare logistic regression with calibrated tree/boosting models and test drift",
                "Add formal feature-attribution methods and adverse-action style reason codes",
                "Document thresholds, approval policy, fairness tests and human-override controls",
                "Containerise, add authentication and secrets management, and create CI/CD",
                "Track model drift, data drift, latency, errors and portfolio outcomes"
            ]
        })
        st.dataframe(roadmap, use_container_width=True, hide_index=True)

    st.markdown("### Viva-ready one-liner")
    st.markdown("<div class='callout'><b>CreditWise AI is a hybrid decision-support app:</b> a transparent ML model estimates credit-screening risk, rule checks provide deterministic guardrails, and Gemini converts the result into a human-readable explanation without being allowed to make the final lending decision.</div>", unsafe_allow_html=True)

    st.markdown("### Sources used in the prototype")
    st.markdown("- Google AI for Developers — Gemini API documentation\n- scikit-learn documentation — Logistic Regression, pipelines and validation metrics\n- Plotly documentation — interactive financial/analytics charts")
