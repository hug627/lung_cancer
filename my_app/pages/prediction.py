import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(
    page_title="LungGuard AI — Risk Assessment",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# ── MINIMAL STYLE OVERRIDES ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1000px; }

.stSelectbox > div > div {
    background: rgba(10,20,35,0.9) !important;
    border: 1px solid rgba(52,152,219,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSelectbox label { color: #94a3b8 !important; font-size: 0.85rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #1a6fa8, #2980b9) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.9rem 3rem !important;
    font-size: 1rem !important; font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(41,128,185,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2980b9, #3498db) !important;
    box-shadow: 0 6px 28px rgba(41,128,185,0.5) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
    background: rgba(10,20,35,0.8);
    border: 1px solid rgba(52,152,219,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"]  { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"]  { font-size: 1.5rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🫁 LungGuard AI — Risk Assessment")
st.caption("Answer the 13 clinical questions below. The model returns a risk score in under a second.")

if api_online:
    st.success("✅ API Live — predictions are enabled.")
else:
    st.error(f"⚠️ Cannot reach the prediction API at `{API_URL}`. Predictions are unavailable.")

st.divider()

# ── HELPER ───────────────────────────────────────────────────────────────────
binary = lambda x: 2 if x == "Yes" else 1

# ── LIFESTYLE FACTORS ─────────────────────────────────────────────────────────
st.markdown("#### 🚬 Lifestyle Factors")
c1, c2, c3 = st.columns(3)
with c1:
    smoking         = st.selectbox("Smoking",          ["No", "Yes"], key="smoking")
    peer_pressure   = st.selectbox("Peer Pressure",    ["No", "Yes"], key="peer")
with c2:
    alcohol         = st.selectbox("Alcohol Consuming",["No", "Yes"], key="alcohol")
    anxiety         = st.selectbox("Anxiety",          ["No", "Yes"], key="anxiety")
with c3:
    yellow_fingers  = st.selectbox("Yellow Fingers",   ["No", "Yes"], key="yellow")
    chronic_disease = st.selectbox("Chronic Disease",  ["No", "Yes"], key="chronic")

st.divider()

# ── PHYSICAL SYMPTOMS ─────────────────────────────────────────────────────────
st.markdown("#### 🩺 Physical Symptoms")
c4, c5, c6, c7 = st.columns(4)
with c4:
    fatigue          = st.selectbox("Fatigue",               ["No", "Yes"], key="fatigue")
    wheezing         = st.selectbox("Wheezing",              ["No", "Yes"], key="wheezing")
with c5:
    coughing         = st.selectbox("Coughing",              ["No", "Yes"], key="coughing")
    allergy          = st.selectbox("Allergy",               ["No", "Yes"], key="allergy")
with c6:
    shortness_breath = st.selectbox("Shortness of Breath",   ["No", "Yes"], key="breath")
    chest_pain       = st.selectbox("Chest Pain",            ["No", "Yes"], key="chest")
with c7:
    swallowing       = st.selectbox("Swallowing Difficulty", ["No", "Yes"], key="swallow")

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICT BUTTON ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Run Risk Assessment", type="primary", disabled=not api_online)

# ── RESULTS ───────────────────────────────────────────────────────────────────
if predict_clicked:
    payload = {
        "SMOKING":               binary(smoking),
        "YELLOW_FINGERS":        binary(yellow_fingers),
        "ANXIETY":               binary(anxiety),
        "PEER_PRESSURE":         binary(peer_pressure),
        "CHRONIC_DISEASE":       binary(chronic_disease),
        "FATIGUE":               binary(fatigue),
        "ALLERGY":               binary(allergy),
        "WHEEZING":              binary(wheezing),
        "ALCOHOL_CONSUMING":     binary(alcohol),
        "COUGHING":              binary(coughing),
        "SHORTNESS_OF_BREATH":   binary(shortness_breath),
        "SWALLOWING_DIFFICULTY": binary(swallowing),
        "CHEST_PAIN":            binary(chest_pain),
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
        is_high     = prediction == "HIGH RISK"

        st.divider()

        # ── Result banner ─────────────────────────────────────────────────────
        if is_high:
            st.error(
                f"### ⚠️ High Risk Detected\n"
                f"The model predicts a **{prob_high:.1%} probability** of lung cancer "
                f"based on the symptoms provided. Immediate medical consultation is strongly recommended."
            )
        else:
            st.success(
                f"### ✅ Low Risk Detected\n"
                f"The model predicts a **{prob_low:.1%} probability** of no lung cancer "
                f"based on the symptoms provided. Continue routine health monitoring."
            )

        # ── Metric boxes ──────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("High Risk Probability", f"{prob_high:.1%}")
        m2.metric("Low Risk Probability",  f"{prob_low:.1%}")
        m3.metric("Model Confidence",      confidence)

        # ── Probability bars ──────────────────────────────────────────────────
        st.markdown("**Risk probability breakdown**")
        st.progress(prob_high, text=f"High Risk — {prob_high:.1%}")
        st.progress(prob_low,  text=f"Low Risk  — {prob_low:.1%}")

        # ── Model validation against dataset ──────────────────────────────────
        if data is not None:
            def make_sample_payload(row):
                return {k: int(row.get(k, 1)) for k in feature_columns}

            pos = data[data['LUNG_CANCER'].isin(['YES', 'yes', 2, 1])]
            neg = data[data['LUNG_CANCER'].isin(['NO',  'no',  0])]

            val_rows = []
            if len(pos) > 0:
                vr, _ = call_predict(make_sample_payload(pos.iloc[0]))
                if vr:
                    val_rows.append(("Known cancer case from dataset", vr["prediction"]))
            if len(neg) > 0:
                vr2, _ = call_predict(make_sample_payload(neg.iloc[0]))
                if vr2:
                    val_rows.append(("Known no-cancer case from dataset", vr2["prediction"]))

            if val_rows:
                st.divider()
                st.markdown("**Model Validation Against Dataset**")
                for label, pred in val_rows:
                    col_a, col_b = st.columns([4, 1])
                    col_a.caption(label)
                    if pred == "HIGH RISK":
                        col_b.error(pred)
                    else:
                        col_b.success(pred)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.divider()
        st.warning(
            "⚠️ **Medical Disclaimer:** "
            + result.get(
                "disclaimer",
                "This prediction is for educational and screening purposes only. "
                "Always consult a qualified medical professional."
            )
        )
