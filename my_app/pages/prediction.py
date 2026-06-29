import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="LungGuard AI — Risk Assessment",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# API SETUP
# ===============================
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

def call_predict(payload: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to API at {API_URL}"
    except requests.exceptions.Timeout:
        return None, "API took too long to respond. Try again."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data
def load_data():
    try:
        return pd.read_csv("my_app/survey lung cancer.csv")
    except Exception:
        return None

data = load_data()
api_online = check_api()

feature_columns = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING",
    "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
]

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0d2137 0%, #0a3d62 100%);
    border-bottom: 1px solid rgba(52,152,219,0.3);
    padding: 2.5rem 3rem 2rem;
    margin: -1rem -1rem 2.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.page-header-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #fff;
    margin: 0 0 0.3rem;
    line-height: 1.2;
}
.page-header-text p { color: #94a3b8; margin: 0; font-size: 0.9rem; }

/* ── API pill ── */
.api-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem 0.9rem; border-radius: 100px;
    font-size: 0.78rem; font-weight: 500;
}
.api-online { background: rgba(39,174,96,0.12); border: 1px solid rgba(39,174,96,0.3); color: #2ecc71; }
.api-offline { background: rgba(231,76,60,0.12); border: 1px solid rgba(231,76,60,0.3); color: #e74c3c; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot-online { background: #2ecc71; box-shadow: 0 0 5px #2ecc71; animation: pulse 2s infinite; }
.dot-offline { background: #e74c3c; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Cards ── */
.card {
    background: rgba(15,30,50,0.6);
    border: 1px solid rgba(52,152,219,0.12);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #5dade2; margin-bottom: 1.2rem;
}

/* ── Symptom toggles ── */
.stSelectbox > div > div {
    background: rgba(10,20,35,0.9) !important;
    border: 1px solid rgba(52,152,219,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSelectbox label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a6fa8, #2980b9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 3rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(41,128,185,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2980b9, #3498db) !important;
    box-shadow: 0 6px 28px rgba(41,128,185,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Result cards ── */
.result-high {
    background: linear-gradient(135deg, rgba(192,57,43,0.15), rgba(231,76,60,0.08));
    border: 1px solid rgba(231,76,60,0.35);
    border-left: 4px solid #e74c3c;
    border-radius: 0 14px 14px 0;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
}
.result-low {
    background: linear-gradient(135deg, rgba(39,174,96,0.12), rgba(46,204,113,0.06));
    border: 1px solid rgba(39,174,96,0.3);
    border-left: 4px solid #2ecc71;
    border-radius: 0 14px 14px 0;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
}
.result-label {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-label-high { color: #e74c3c; }
.result-label-low { color: #2ecc71; }
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem; font-weight: 700;
    color: #fff; margin-bottom: 0.5rem;
}
.result-subtitle { color: #94a3b8; font-size: 0.88rem; line-height: 1.6; }

/* ── Probability bar ── */
.prob-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 100px; height: 10px;
    margin: 0.5rem 0 0.3rem; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 100px;
    transition: width 0.8s ease;
}
.prob-bar-high { background: linear-gradient(90deg, #c0392b, #e74c3c); }
.prob-bar-low  { background: linear-gradient(90deg, #27ae60, #2ecc71); }
.prob-label { font-size: 0.78rem; color: #64748b; margin-bottom: 0.8rem; }

/* ── Metric boxes ── */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
.metric-box {
    background: rgba(10,20,35,0.8);
    border: 1px solid rgba(52,152,219,0.12);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700; color: #5dade2; }
.metric-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

/* ── Validation ── */
.val-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.7rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.83rem;
}
.val-dot-pos { color: #2ecc71; font-size: 1rem; }
.val-dot-neg { color: #5dade2; font-size: 1rem; }
.val-text { color: #94a3b8; }
.val-badge {
    margin-left: auto; font-size: 0.72rem; font-weight: 600;
    padding: 0.2rem 0.7rem; border-radius: 6px;
}
.badge-high { background: rgba(231,76,60,0.15); color: #e74c3c; }
.badge-low  { background: rgba(39,174,96,0.15); color: #2ecc71; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(231,76,60,0.05);
    border: 1px solid rgba(231,76,60,0.15);
    border-radius: 10px; padding: 1rem 1.3rem;
    color: #94a3b8; font-size: 0.78rem; line-height: 1.7;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# PAGE HEADER
# ===============================
st.markdown(f"""
<div class="page-header">
  <div style="font-size:3rem;">🫁</div>
  <div class="page-header-text">
    <h1>LungGuard AI — Risk Assessment</h1>
    <p>Answer the 13 clinical questions below. The model returns a risk score in under a second.</p>
  </div>
  <div style="margin-left:auto;">
    <span class="api-pill {'api-online' if api_online else 'api-offline'}">
      <span class="dot {'dot-online' if api_online else 'dot-offline'}"></span>
      {'API Live' if api_online else 'API Offline'}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

if not api_online:
    st.error(f"⚠️ Cannot reach the prediction API at `{API_URL}`. Predictions are unavailable until it's back online.")

# ===============================
# INPUT FORM
# ===============================
binary = lambda x: 2 if x == "Yes" else 1

st.markdown('<div class="card"><div class="card-title">🚬 Lifestyle Factors</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    smoking = st.selectbox("Smoking", ["No", "Yes"], key="smoking")
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"], key="peer")
with c2:
    alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"], key="alcohol")
    anxiety = st.selectbox("Anxiety", ["No", "Yes"], key="anxiety")
with c3:
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"], key="yellow")
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"], key="chronic")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">🩺 Physical Symptoms</div>', unsafe_allow_html=True)
c4, c5, c6, c7 = st.columns(4)
with c4:
    fatigue = st.selectbox("Fatigue", ["No", "Yes"], key="fatigue")
    wheezing = st.selectbox("Wheezing", ["No", "Yes"], key="wheezing")
with c5:
    coughing = st.selectbox("Coughing", ["No", "Yes"], key="coughing")
    allergy = st.selectbox("Allergy", ["No", "Yes"], key="allergy")
with c6:
    shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"], key="breath")
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"], key="chest")
with c7:
    swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"], key="swallow")
st.markdown('</div>', unsafe_allow_html=True)

# Input preview
with st.expander("🔎 View encoded input data"):
    payload_preview = {
        "SMOKING": binary(smoking), "YELLOW_FINGERS": binary(yellow_fingers),
        "ANXIETY": binary(anxiety), "PEER_PRESSURE": binary(peer_pressure),
        "CHRONIC_DISEASE": binary(chronic_disease), "FATIGUE": binary(fatigue),
        "ALLERGY": binary(allergy), "WHEEZING": binary(wheezing),
        "ALCOHOL_CONSUMING": binary(alcohol), "COUGHING": binary(coughing),
        "SHORTNESS_OF_BREATH": binary(shortness_breath),
        "SWALLOWING_DIFFICULTY": binary(swallowing), "CHEST_PAIN": binary(chest_pain),
    }
    st.dataframe(pd.DataFrame([payload_preview]), use_container_width=True)
    st.caption("Encoding: 1 = No, 2 = Yes")

# ===============================
# PREDICT BUTTON
# ===============================
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Run Risk Assessment", type="primary", disabled=not api_online)

# ===============================
# RESULTS
# ===============================
if predict_clicked:
    payload = {
        "SMOKING": binary(smoking), "YELLOW_FINGERS": binary(yellow_fingers),
        "ANXIETY": binary(anxiety), "PEER_PRESSURE": binary(peer_pressure),
        "CHRONIC_DISEASE": binary(chronic_disease), "FATIGUE": binary(fatigue),
        "ALLERGY": binary(allergy), "WHEEZING": binary(wheezing),
        "ALCOHOL_CONSUMING": binary(alcohol), "COUGHING": binary(coughing),
        "SHORTNESS_OF_BREATH": binary(shortness_breath),
        "SWALLOWING_DIFFICULTY": binary(swallowing), "CHEST_PAIN": binary(chest_pain),
    }

    with st.spinner("Analysing symptoms via API..."):
        result, error = call_predict(payload)

    if error:
        st.error(f"Prediction failed: {error}")
    else:
        prediction  = result["prediction"]
        prob_high   = result["probability_high_risk"]
        prob_low    = result["probability_low_risk"]
        confidence  = result["model_confidence"]
        pred_code   = result["prediction_code"]

        is_high = prediction == "HIGH RISK"

        # ── Main result banner ──
        if is_high:
            st.markdown(f"""
            <div class="result-high">
              <div class="result-label result-label-high">⚠️ Assessment Result</div>
              <div class="result-title">High Risk Detected</div>
              <div class="result-subtitle">
                The model predicts a <strong style="color:#e74c3c">{prob_high:.1%} probability</strong> of lung cancer 
                based on the symptoms provided. Immediate medical consultation is strongly recommended.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <div class="result-label result-label-low">✅ Assessment Result</div>
              <div class="result-title">Low Risk Detected</div>
              <div class="result-subtitle">
                The model predicts a <strong style="color:#2ecc71">{prob_low:.1%} probability</strong> of no lung cancer 
                based on the symptoms provided. Continue routine health monitoring.
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Metric boxes ──
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-box">
            <div class="metric-val" style="color:{'#e74c3c' if is_high else '#2ecc71'}">{prob_high:.1%}</div>
            <div class="metric-lbl">High Risk Probability</div>
          </div>
          <div class="metric-box">
            <div class="metric-val">{prob_low:.1%}</div>
            <div class="metric-lbl">Low Risk Probability</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" style="color:#5dade2">{confidence}</div>
            <div class="metric-lbl">Model Confidence</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown(f"""
        <div style="margin:1.5rem 0;">
          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
            <span style="font-size:0.8rem; color:#94a3b8;">High Risk</span>
            <span style="font-size:0.8rem; color:#e74c3c; font-weight:600;">{prob_high:.1%}</span>
          </div>
          <div class="prob-bar-wrap"><div class="prob-bar-fill prob-bar-high" style="width:{prob_high*100:.1f}%"></div></div>
          <div style="display:flex; justify-content:space-between; margin: 1rem 0 0.3rem;">
            <span style="font-size:0.8rem; color:#94a3b8;">Low Risk</span>
            <span style="font-size:0.8rem; color:#2ecc71; font-weight:600;">{prob_low:.1%}</span>
          </div>
          <div class="prob-bar-wrap"><div class="prob-bar-fill prob-bar-low" style="width:{prob_low*100:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model Validation ──
        if data is not None:
            st.markdown('<div class="card"><div class="card-title">✅ Model Validation Against Dataset</div>', unsafe_allow_html=True)

            def make_sample_payload(row):
                return {k: int(row.get(k, 1)) for k in feature_columns}

            pos = data[data['LUNG_CANCER'].isin(['YES', 'yes', 2, 1])]
            neg = data[data['LUNG_CANCER'].isin(['NO', 'no', 0])]

            val_html = ""
            if len(pos) > 0:
                vr, _ = call_predict(make_sample_payload(pos.iloc[0]))
                if vr:
                    badge = "badge-high" if vr['prediction'] == "HIGH RISK" else "badge-low"
                    val_html += f"""<div class="val-row">
                        <span class="val-dot-pos">●</span>
                        <span class="val-text">Known cancer case from dataset</span>
                        <span class="val-badge {badge}">{vr['prediction']}</span>
                    </div>"""

            if len(neg) > 0:
                vr2, _ = call_predict(make_sample_payload(neg.iloc[0]))
                if vr2:
                    badge2 = "badge-high" if vr2['prediction'] == "HIGH RISK" else "badge-low"
                    val_html += f"""<div class="val-row">
                        <span class="val-dot-neg">●</span>
                        <span class="val-text">Known no-cancer case from dataset</span>
                        <span class="val-badge {badge2}">{vr2['prediction']}</span>
                    </div>"""

            st.markdown(val_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Disclaimer ──
        st.markdown(f"""
        <div class="disclaimer">
          ⚠️ <strong>Medical Disclaimer:</strong> {result.get('disclaimer', 
          'This prediction is for educational and screening purposes only. Always consult a qualified medical professional.')}
        </div>
        """, unsafe_allow_html=True)
