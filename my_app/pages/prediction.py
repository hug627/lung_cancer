import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Cancer Prediction", layout="wide")
st.title("🤖 Lung Cancer Prediction System")

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
        return None, f"❌ Cannot connect to API at {API_URL}"
    except requests.exceptions.Timeout:
        return None, "⏱️ API took too long to respond. Try again."
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

# ===============================
# LOAD DATA (for sidebar + validation)
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("my_app/survey lung cancer.csv")
    except Exception:
        return None

data = load_data()

feature_columns = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING",
    "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("API Info")
api_online = check_api()

if api_online:
    st.sidebar.success("API Online ✅")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=5).json()
        st.sidebar.write(f"**Model:** {info.get('model_type', 'Logistic Regression')}")
        st.sidebar.write(f"**CV Accuracy:** {info.get('cross_validation_accuracy', '91.9%')}")
        st.sidebar.write(f"**Features:** {info.get('num_features', 13)}")
    except:
        pass
else:
    st.sidebar.error("API Offline ❌")
    st.sidebar.caption(f"Expected at:\n`{API_URL}`")

if data is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"Total features: {len(feature_columns)}")
    st.sidebar.write("Features used:", feature_columns)
    st.sidebar.write("Sample data:")
    st.sidebar.dataframe(data.head(3))

st.sidebar.markdown("---")
st.sidebar.header("Encoding Settings")
st.sidebar.radio("Select encoding:", ["YES=2, NO=1"], index=0)

# ===============================
# API STATUS BANNER
# ===============================
if api_online:
    st.success(f"✅ Connected to prediction API → `{API_URL}`")
else:
    st.error(f"❌ API not reachable at `{API_URL}` — predictions will fail.")

# ===============================
# INPUT FORM — 13 features only (matches your trained model)
# ===============================
st.header("🧪 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox("Smoking", ["No", "Yes"], key="smoking")
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"], key="yellow")
    anxiety = st.selectbox("Anxiety", ["No", "Yes"], key="anxiety")
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"], key="peer")
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"], key="chronic")
    fatigue = st.selectbox("Fatigue", ["No", "Yes"], key="fatigue")
    allergy = st.selectbox("Allergy", ["No", "Yes"], key="allergy")

with col2:
    wheezing = st.selectbox("Wheezing", ["No", "Yes"], key="wheezing")
    alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"], key="alcohol")
    coughing = st.selectbox("Coughing", ["No", "Yes"], key="coughing")
    shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"], key="breath")
    swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"], key="swallow")
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"], key="chest")

# ===============================
# BUILD PAYLOAD — 13 features, no GENDER/AGE
# ===============================
binary = lambda x: 2 if x == "Yes" else 1

payload = {
    "SMOKING": binary(smoking),
    "YELLOW_FINGERS": binary(yellow_fingers),
    "ANXIETY": binary(anxiety),
    "PEER_PRESSURE": binary(peer_pressure),
    "CHRONIC_DISEASE": binary(chronic_disease),
    "FATIGUE": binary(fatigue),
    "ALLERGY": binary(allergy),
    "WHEEZING": binary(wheezing),
    "ALCOHOL_CONSUMING": binary(alcohol),
    "COUGHING": binary(coughing),
    "SHORTNESS_OF_BREATH": binary(shortness_breath),
    "SWALLOWING_DIFFICULTY": binary(swallowing),
    "CHEST_PAIN": binary(chest_pain),
}

# Show input — same as your original expander
with st.expander("View Input Data"):
    input_df = pd.DataFrame([payload])
    st.dataframe(input_df)
    st.write(f"Shape: {input_df.shape}")
    st.write(f"Columns: {input_df.columns.tolist()}")

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict", type="primary"):
    if not api_online:
        st.error("Cannot predict — API is offline.")
    else:
        with st.spinner("Getting prediction from API..."):
            result, error = call_predict(payload)

        if error:
            st.error(f"Prediction failed: {error}")
            with st.expander("Show error details"):
                st.code(error)
        else:
            st.write("---")
            st.subheader("Results")

            prediction = result["prediction"]
            prob_high = result["probability_high_risk"]
            prob_low = result["probability_low_risk"]
            confidence = result["model_confidence"]

            st.write(f"**Raw prediction value:** `{result['prediction_code']}` → `{prediction}`")
            st.write(f"**Probabilities:** High Risk={prob_high:.4f}, Low Risk={prob_low:.4f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Class 0 (No Cancer)", f"{prob_low:.1%}")
            with col_b:
                st.metric("Class 1 (Cancer)", f"{prob_high:.1%}")

            if prediction == "HIGH RISK":
                st.error("### ⚠️ HIGH RISK: Lung Cancer Detected")
            else:
                st.success("### ✅ LOW RISK: No Lung Cancer Detected")

            st.info("⚠️ This is for educational purposes only. Consult a healthcare professional.")

            # Model Validation
            if data is not None:
                st.write("---")
                st.subheader("Model Validation")

                def make_sample_payload(row):
                    return {
                        "SMOKING": int(row.get("SMOKING", 1)),
                        "YELLOW_FINGERS": int(row.get("YELLOW_FINGERS", 1)),
                        "ANXIETY": int(row.get("ANXIETY", 1)),
                        "PEER_PRESSURE": int(row.get("PEER_PRESSURE", 1)),
                        "CHRONIC_DISEASE": int(row.get("CHRONIC_DISEASE", 1)),
                        "FATIGUE": int(row.get("FATIGUE", 1)),
                        "ALLERGY": int(row.get("ALLERGY", 1)),
                        "WHEEZING": int(row.get("WHEEZING", 1)),
                        "ALCOHOL_CONSUMING": int(row.get("ALCOHOL_CONSUMING", 1)),
                        "COUGHING": int(row.get("COUGHING", 1)),
                        "SHORTNESS_OF_BREATH": int(row.get("SHORTNESS_OF_BREATH", 1)),
                        "SWALLOWING_DIFFICULTY": int(row.get("SWALLOWING_DIFFICULTY", 1)),
                        "CHEST_PAIN": int(row.get("CHEST_PAIN", 1)),
                    }

                positive_samples = data[data['LUNG_CANCER'].isin(['YES', 'yes', 2, 1])]
                if len(positive_samples) > 0:
                    val_result, _ = call_predict(make_sample_payload(positive_samples.iloc[0]))
                    if val_result:
                        st.write(f"✓ Test on known cancer case: Prediction = `{val_result['prediction_code']}` ({val_result['prediction']})")

                negative_samples = data[data['LUNG_CANCER'].isin(['NO', 'no', 0])]
                if len(negative_samples) > 0:
                    val_result_no, _ = call_predict(make_sample_payload(negative_samples.iloc[0]))
                    if val_result_no:
                        st.write(f"✓ Test on known no-cancer case: Prediction = `{val_result_no['prediction_code']}` ({val_result_no['prediction']})")

            st.caption(result.get("disclaimer", ""))
