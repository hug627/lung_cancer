import streamlit as st
import requests
import os

st.set_page_config(
    page_title="LungGuard AI — Early Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False

api_online = check_api()

# ── MINIMAL STYLE OVERRIDES ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1000px; }

.stButton > button {
    background: linear-gradient(135deg, #1a6fa8, #2980b9) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.85rem 2.5rem !important;
    font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(41,128,185,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2980b9, #3498db) !important;
    box-shadow: 0 6px 28px rgba(41,128,185,0.55) !important;
    transform: translateY(-2px) !important;
}

/* metric cards */
[data-testid="stMetric"] {
    background: rgba(10,20,35,0.8);
    border: 1px solid rgba(52,152,219,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #5dade2 !important; font-size: 1.7rem !important; font-weight: 700 !important; }

/* info / warning / success boxes */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* expanders */
[data-testid="stExpander"] {
    background: rgba(15,30,50,0.6) !important;
    border: 1px solid rgba(52,152,219,0.12) !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("# 🫁 LungGuard AI")
st.markdown("#### AI-Powered Lung Cancer Early Detection")
st.caption(
    "A symptom-based screening system that analyzes 13 clinical risk factors "
    "and lifestyle signals to assess lung cancer risk — powered by machine learning."
)

if api_online:
    st.success("✅ Prediction API is online and ready.")
else:
    st.error(f"⚠️ Prediction API is offline ({API_URL}). Risk assessment is unavailable.")

st.divider()

# ── STATS ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("CV Accuracy", "91.9%")
c2.metric("Clinical Features", "13")
c3.metric("Training Records", "309")
c4.metric("Models Evaluated", "5")

st.divider()

# ── CTA ───────────────────────────────────────────────────────────────────────
_, mid, _ = st.columns([2, 1, 2])
with mid:
    if st.button("🔍 Run Risk Assessment →", type="primary", disabled=not api_online):
        st.switch_page("pages/2_Prediction.py")

st.divider()

# ── SECTION 1 — PURPOSE ───────────────────────────────────────────────────────
with st.expander("01 — What This System Does", expanded=True):
    st.markdown("""
LungGuard AI identifies whether a patient may be at risk of lung cancer based on their
lifestyle factors and reported symptoms. It is designed as a **cost-effective first-line
screening tool** — helping healthcare workers prioritise patients for clinical investigation
before expensive scans are ordered.

The system is trained on the **Survey Lung Cancer Dataset**, which captures demographic data,
lifestyle habits, and physical symptoms alongside confirmed lung cancer diagnoses.
""")

# ── SECTION 2 — FEATURES ─────────────────────────────────────────────────────
with st.expander("02 — 13 Clinical Features Analyzed"):
    features = [
        ("🚬", "Smoking",               "History of tobacco use — the strongest known risk factor for lung cancer."),
        ("🖐️", "Yellow Fingers",         "Nicotine staining — a visible marker of long-term heavy smoking."),
        ("😰", "Anxiety",               "Chronic anxiety linked to stress-related immune suppression."),
        ("👥", "Peer Pressure",          "Social influence leading to adoption of harmful habits."),
        ("🏥", "Chronic Disease",        "Pre-existing conditions that may interact with cancer risk."),
        ("😴", "Fatigue",               "Persistent exhaustion — a common early oncological symptom."),
        ("🤧", "Allergy",               "Chronic allergic responses affecting respiratory health."),
        ("💨", "Wheezing",              "Whistling breath indicating airway obstruction or narrowing."),
        ("🍺", "Alcohol Consuming",      "Regular alcohol use as a compounding lifestyle risk factor."),
        ("🫁", "Coughing",              "Persistent cough — one of the most common lung cancer signals."),
        ("😮‍💨", "Shortness of Breath",   "Dyspnea indicating reduced lung capacity or obstruction."),
        ("🍽️", "Swallowing Difficulty",  "Dysphagia — may indicate tumour pressure on the oesophagus."),
        ("💔", "Chest Pain",            "Persistent chest discomfort — a key clinical warning sign."),
    ]
    for icon, name, desc in features:
        col_a, col_b = st.columns([1, 8])
        with col_a:
            st.markdown(f"### {icon}")
        with col_b:
            st.markdown(f"**{name}**  \n{desc}")
        st.divider()

# ── SECTION 3 — CONTEXT ───────────────────────────────────────────────────────
with st.expander("03 — Why Early Detection Matters"):
    st.warning(
        "Lung cancer is the **leading cause of cancer-related deaths worldwide**, "
        "yet survival rates improve dramatically when caught early."
    )
    steps = [
        ("Traditional Screening Gaps",
         "CT scans and biopsies are invasive, expensive, and unavailable to many populations — "
         "especially in resource-limited healthcare settings."),
        ("The Symptom Blind Spot",
         "Many patients dismiss early warning signs as minor issues, delaying consultation "
         "until the cancer has advanced significantly."),
        ("LungGuard's Role",
         "A fast, free, symptom-based risk signal that prompts earlier clinical consultation — "
         "before expensive diagnostics are needed."),
        ("Optimised for Sensitivity",
         "The model prioritises high recall — catching as many true cases as possible, "
         "minimising dangerous false negatives."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"**{i}. {title}**")
        st.caption(body)
        if i < len(steps):
            st.divider()

# ── SECTION 4 — METHODOLOGY ───────────────────────────────────────────────────
with st.expander("04 — How the Prediction Works"):
    st.markdown("""
Five machine learning models were trained and evaluated. **Logistic Regression** was selected
for its interpretability and strong recall performance.
""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("✓ **Logistic Regression** ← Selected model")
        st.info("Support Vector Machine")
        st.info("Random Forest")
    with col2:
        st.info("XGBoost")
        st.info("Neural Network (MLP)")

    st.markdown("""
The model is deployed via a **FastAPI backend on Hugging Face Spaces**.
Every prediction is a live API call — no model is loaded locally in the browser.
""")

# ── SECTION 5 — IMPACT ───────────────────────────────────────────────────────
with st.expander("05 — What We Aim to Achieve"):
    impacts = [
        ("🎯", "Earlier Detection",   "Flag high-risk patients before symptoms become severe."),
        ("💰", "Lower Costs",         "Symptom-based screening before expensive imaging procedures."),
        ("⏱️", "Faster Triage",       "Instant risk scores to help clinicians prioritise queues."),
        ("📊", "Research Insights",   "Surface patterns between lifestyle factors and cancer diagnosis."),
        ("❤️", "Save Lives",          "Earlier intervention leads to better treatment outcomes."),
    ]
    cols = st.columns(len(impacts))
    for col, (icon, title, body) in zip(cols, impacts):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(body)

st.divider()

# ── DISCLAIMER ────────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Medical Disclaimer:** LungGuard AI is a screening and educational tool only. "
    "It does not constitute medical advice, diagnosis, or treatment. All risk assessments "
    "must be confirmed by a qualified healthcare professional. "
    "If you are experiencing symptoms, seek medical attention immediately."
)
st.info(
    "ℹ️ **About This Project:** Built by Mercy Mwova as a portfolio project demonstrating "
    "end-to-end ML deployment — from data science (Python, scikit-learn) to production API "
    "(FastAPI, Hugging Face Spaces) to live frontend (Streamlit Cloud).  \n"
    "[View on GitHub →](https://github.com/hug627/lung_cancer)"
)
