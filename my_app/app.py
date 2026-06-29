import streamlit as st
import requests
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="LungGuard AI — Early Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# API STATUS
# ===============================
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False

api_online = check_api()

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 0rem !important;}

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #0a3d62 50%, #1a5276 100%);
    border-bottom: 1px solid rgba(52, 152, 219, 0.3);
    padding: 4rem 3rem 3rem;
    margin: -1rem -1rem 3rem -1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(52,152,219,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(52, 152, 219, 0.15);
    border: 1px solid rgba(52, 152, 219, 0.4);
    color: #5dade2;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: #5dade2;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 620px;
    margin: 0 auto 2rem;
    line-height: 1.7;
    font-weight: 300;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.hero-stat {
    text-align: center;
}
.hero-stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #5dade2;
    display: block;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── API Status pill ── */
.api-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 2rem;
}
.api-online {
    background: rgba(39, 174, 96, 0.12);
    border: 1px solid rgba(39, 174, 96, 0.3);
    color: #2ecc71;
}
.api-offline {
    background: rgba(231, 76, 60, 0.12);
    border: 1px solid rgba(231, 76, 60, 0.3);
    color: #e74c3c;
}
.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.dot-online { background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }
.dot-offline { background: #e74c3c; }

/* ── Section cards ── */
.section-card {
    background: rgba(15, 30, 50, 0.6);
    border: 1px solid rgba(52, 152, 219, 0.12);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5dade2;
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1rem;
    line-height: 1.3;
}
.section-body {
    color: #94a3b8;
    line-height: 1.8;
    font-size: 0.95rem;
}

/* ── Feature grid ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
}
.feature-card {
    background: rgba(10, 20, 35, 0.8);
    border: 1px solid rgba(52, 152, 219, 0.1);
    border-radius: 12px;
    padding: 1.4rem;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: rgba(52, 152, 219, 0.35); }
.feature-icon {
    font-size: 1.6rem;
    margin-bottom: 0.7rem;
    display: block;
}
.feature-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.feature-desc {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.5;
}

/* ── Timeline steps ── */
.timeline {
    position: relative;
    padding-left: 2rem;
}
.timeline::before {
    content: '';
    position: absolute;
    left: 0.6rem;
    top: 0.5rem;
    bottom: 0.5rem;
    width: 1px;
    background: linear-gradient(to bottom, #3498db, transparent);
}
.timeline-step {
    position: relative;
    padding: 0 0 1.8rem 1.5rem;
}
.timeline-step::before {
    content: '';
    position: absolute;
    left: -1.4rem;
    top: 0.35rem;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #3498db;
    box-shadow: 0 0 8px rgba(52,152,219,0.5);
}
.timeline-step-title {
    font-weight: 600;
    color: #e2e8f0;
    font-size: 0.92rem;
    margin-bottom: 0.25rem;
}
.timeline-step-body {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.6;
}

/* ── Impact stat cards ── */
.impact-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.impact-card {
    background: rgba(10, 20, 35, 0.8);
    border: 1px solid rgba(52, 152, 219, 0.1);
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
}
.impact-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.impact-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #5dade2;
    margin-bottom: 0.4rem;
}
.impact-body { font-size: 0.78rem; color: #64748b; line-height: 1.5; }

/* ── Models list ── */
.model-tag {
    display: inline-block;
    background: rgba(52, 152, 219, 0.1);
    border: 1px solid rgba(52, 152, 219, 0.2);
    color: #5dade2;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    margin: 0.2rem;
}

/* ── Disclaimer box ── */
.disclaimer-box {
    background: rgba(231, 76, 60, 0.06);
    border: 1px solid rgba(231, 76, 60, 0.2);
    border-left: 3px solid #e74c3c;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-top: 2rem;
    color: #e2e8f0;
    font-size: 0.85rem;
    line-height: 1.7;
}
.about-box {
    background: rgba(52, 152, 219, 0.06);
    border: 1px solid rgba(52, 152, 219, 0.2);
    border-left: 3px solid #3498db;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.7;
}

/* ── Scroll reveal ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
.section-card { animation: fadeUp 0.5s ease both; }
</style>
""", unsafe_allow_html=True)

# ===============================
# HERO
# ===============================
api_status_html = f"""
<div style="text-align:center; margin-bottom:0.5rem;">
  <span class="api-pill {'api-online' if api_online else 'api-offline'}">
    <span class="dot {'dot-online' if api_online else 'dot-offline'}"></span>
    {'Prediction API Online' if api_online else 'Prediction API Offline'}
  </span>
</div>
"""

st.markdown(f"""
<div class="hero">
  <div class="hero-badge">🫁 AI-Powered Healthcare Screening</div>
  <div class="hero-title">Lung<span>Guard</span> AI</div>
  <div class="hero-subtitle">
    An early-detection screening system that analyzes clinical symptoms and lifestyle 
    factors to assess lung cancer risk — powered by machine learning.
  </div>
  {api_status_html}
  <div class="hero-stats">
    <div class="hero-stat">
      <span class="hero-stat-number">91.9%</span>
      <div class="hero-stat-label">CV Accuracy</div>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-number">13</span>
      <div class="hero-stat-label">Clinical Features</div>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-number">309</span>
      <div class="hero-stat-label">Training Records</div>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-number">5</span>
      <div class="hero-stat-label">Models Evaluated</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SECTION 1 — Objective
# ===============================
st.markdown("""
<div class="section-card">
  <div class="section-label">01 — Purpose</div>
  <div class="section-title">What This System Does</div>
  <div class="section-body">
    LungGuard AI is a machine learning prediction system that identifies whether a patient 
    may have lung cancer based on their lifestyle factors and reported symptoms. It is designed 
    as a <strong style="color:#5dade2">cost-effective first-line screening tool</strong> — helping 
    healthcare workers prioritize patients for clinical investigation without expensive scans upfront.
    <br><br>
    The system analyzes the <strong style="color:#e2e8f0">Survey Lung Cancer Dataset</strong>, 
    which captures demographic data, lifestyle habits, and physical symptoms alongside confirmed 
    lung cancer diagnoses.
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SECTION 2 — Features
# ===============================
st.markdown("""
<div class="section-card">
  <div class="section-label">02 — Inputs</div>
  <div class="section-title">13 Clinical Features Analyzed</div>
  <div class="section-body">The model evaluates these symptom and lifestyle signals:</div>
  <div class="feature-grid">
    <div class="feature-card">
      <span class="feature-icon">🚬</span>
      <div class="feature-name">Smoking</div>
      <div class="feature-desc">History of tobacco use — the strongest known risk factor for lung cancer.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">🖐️</span>
      <div class="feature-name">Yellow Fingers</div>
      <div class="feature-desc">Nicotine staining — a visible marker of long-term heavy smoking.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">😰</span>
      <div class="feature-name">Anxiety</div>
      <div class="feature-desc">Chronic anxiety linked to stress-related immune suppression.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">👥</span>
      <div class="feature-name">Peer Pressure</div>
      <div class="feature-desc">Social influence leading to adoption of harmful habits.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">🏥</span>
      <div class="feature-name">Chronic Disease</div>
      <div class="feature-desc">Pre-existing conditions that may interact with cancer risk.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">😴</span>
      <div class="feature-name">Fatigue</div>
      <div class="feature-desc">Persistent exhaustion — a common early oncological symptom.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">🤧</span>
      <div class="feature-name">Allergy</div>
      <div class="feature-desc">Chronic allergic responses affecting respiratory health.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">💨</span>
      <div class="feature-name">Wheezing</div>
      <div class="feature-desc">Whistling breath indicating airway obstruction or narrowing.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">🍺</span>
      <div class="feature-name">Alcohol Consuming</div>
      <div class="feature-desc">Regular alcohol use as a compounding lifestyle risk factor.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">🫁</span>
      <div class="feature-name">Coughing</div>
      <div class="feature-desc">Persistent cough — one of the most common lung cancer signals.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">😮‍💨</span>
      <div class="feature-name">Shortness of Breath</div>
      <div class="feature-desc">Dyspnea indicating reduced lung capacity or obstruction.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">🍽️</span>
      <div class="feature-name">Swallowing Difficulty</div>
      <div class="feature-desc">Dysphagia — may indicate tumour pressure on the oesophagus.</div>
    </div>
    <div class="feature-card">
      <span class="feature-icon">💔</span>
      <div class="feature-name">Chest Pain</div>
      <div class="feature-desc">Persistent chest discomfort — a key clinical warning sign.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SECTION 3 — Context
# ===============================
st.markdown("""
<div class="section-card">
  <div class="section-label">03 — Context</div>
  <div class="section-title">Why Early Detection Matters</div>
  <div class="section-body">
    Lung cancer is the <strong style="color:#e74c3c">leading cause of cancer-related deaths worldwide</strong>, 
    yet survival rates improve dramatically when caught early. The challenge: most cases are 
    diagnosed at advanced stages when treatment options are limited and expensive.
  </div>
  <br>
  <div class="timeline">
    <div class="timeline-step">
      <div class="timeline-step-title">The Problem with Traditional Screening</div>
      <div class="timeline-step-body">CT scans and biopsies are invasive, expensive, and unavailable to many populations — especially in resource-limited healthcare settings.</div>
    </div>
    <div class="timeline-step">
      <div class="timeline-step-title">The Symptom Blind Spot</div>
      <div class="timeline-step-body">Many patients dismiss early warning signs as minor issues, delaying consultation until the cancer has advanced significantly.</div>
    </div>
    <div class="timeline-step">
      <div class="timeline-step-title">LungGuard's Role</div>
      <div class="timeline-step-body">This system provides a fast, free, symptom-based risk signal that can prompt earlier clinical consultation — before expensive diagnostics are needed.</div>
    </div>
    <div class="timeline-step">
      <div class="timeline-step-title">Goals: Sensitivity First</div>
      <div class="timeline-step-body">The model is optimised for high recall — catching as many true cases as possible, minimising dangerous false negatives.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SECTION 4 — How It Works
# ===============================
st.markdown("""
<div class="section-card">
  <div class="section-label">04 — Methodology</div>
  <div class="section-title">How the Prediction Works</div>
  <div class="section-body">
    Five machine learning models were trained and evaluated on the dataset. 
    Logistic Regression was selected for its interpretability and strong recall performance.
  </div>
  <br>
  <div style="margin-bottom:1rem;">
    <span class="model-tag">✓ Logistic Regression</span>
    <span class="model-tag">Support Vector Machine</span>
    <span class="model-tag">Random Forest</span>
    <span class="model-tag">XGBoost</span>
    <span class="model-tag">Neural Network (MLP)</span>
  </div>
  <div class="section-body">
    The selected model is deployed via a <strong style="color:#5dade2">FastAPI backend on Hugging Face Spaces</strong>, 
    which this Streamlit interface calls in real time. Every prediction is a live API call — 
    no model is loaded locally.
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SECTION 5 — Impact
# ===============================
st.markdown("""
<div class="section-card">
  <div class="section-label">05 — Impact</div>
  <div class="section-title">What We Aim to Achieve</div>
  <div class="impact-grid">
    <div class="impact-card">
      <div class="impact-icon">🎯</div>
      <div class="impact-title">Earlier Detection</div>
      <div class="impact-body">Flag high-risk patients before symptoms become severe or irreversible.</div>
    </div>
    <div class="impact-card">
      <div class="impact-icon">💰</div>
      <div class="impact-title">Lower Costs</div>
      <div class="impact-body">Symptom-based screening before expensive imaging or biopsy procedures.</div>
    </div>
    <div class="impact-card">
      <div class="impact-icon">⏱️</div>
      <div class="impact-title">Faster Triage</div>
      <div class="impact-body">Instant risk scores to help clinicians prioritise patient queues.</div>
    </div>
    <div class="impact-card">
      <div class="impact-icon">📊</div>
      <div class="impact-title">Research Insights</div>
      <div class="impact-body">Surface patterns between lifestyle factors and cancer diagnosis.</div>
    </div>
    <div class="impact-card">
      <div class="impact-icon">❤️</div>
      <div class="impact-title">Save Lives</div>
      <div class="impact-body">Earlier intervention leads directly to better treatment outcomes.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# DISCLAIMER
# ===============================
st.markdown("""
<div class="disclaimer-box">
  <strong>⚠️ Medical Disclaimer</strong><br>
  LungGuard AI is a screening and educational tool only. It does not constitute medical advice, 
  diagnosis, or treatment. All risk assessments must be confirmed by a qualified healthcare 
  professional through proper clinical examination and diagnostic procedures. If you are experiencing 
  symptoms, seek medical attention immediately.
</div>
<div class="about-box">
  <strong>ℹ️ About This Project</strong><br>
  Built by <strong>Mercy Njoki</strong> as a portfolio project demonstrating end-to-end ML deployment — 
  from data science (Python, scikit-learn) to production API (FastAPI, Hugging Face Spaces) 
  to live frontend (Streamlit Cloud). 
  <a href="https://github.com/hug627/lung_cancer" style="color:#5dade2;">View on GitHub →</a>
</div>
<br>
""", unsafe_allow_html=True)
