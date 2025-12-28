import streamlit as st   
import pandas as pd   
import numpy as np 
import joblib 

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Cancer Prediction", layout="wide")
st.title("🤖 Lung Cancer Prediction System")

# ===============================
# LOAD DATA FIRST
# ===============================
@st.cache_data
def load_data():
    try:
        csv_file ="lung_cancer/my_app/survey lung cancer.csv"
        return pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Cannot load dataset: {e}")
        return None

data = load_data()

if data is None:
    st.stop()

# Get feature columns (exclude LUNG_CANCER, AGE, GENDER)
feature_columns = [col for col in data.columns if col not in ['LUNG_CANCER', 'AGE', 'GENDER']]

st.sidebar.header("Dataset Info")
st.sidebar.write(f"Total features: {len(feature_columns)}")
st.sidebar.write("Features used:", feature_columns)

# Show sample encoding
st.sidebar.write("Sample data:")
st.sidebar.dataframe(data.head(3))

# ===============================
# LOAD MODELS
# ===============================
st.header("📁 Load Model")

# List all .pkl files in directory
model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]

if not model_files:
    st.error("❌ No .pkl model files found in current directory!")
    st.info("Make sure your model files (svm.pkl, rf.pkl, etc.) are in the same folder as this script")
    st.stop()

st.write("Available model files:", model_files)

# Let user select model file
selected_file = st.selectbox("Select model file:", model_files)

try:
    model = joblib.load(selected_file)
    st.success(f"✅ Model loaded: {selected_file}")
    
    # Show model info
    if hasattr(model, 'n_features_in_'):
        st.info(f"Model expects {model.n_features_in_} features")
        if model.n_features_in_ != len(feature_columns):
            st.error(f"⚠️ Mismatch! Model expects {model.n_features_in_} but we have {len(feature_columns)} features")
            
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===============================
# INPUT FORM
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
# ENCODING OPTIONS
# ===============================
st.sidebar.header("Encoding Settings")
encoding_type = st.sidebar.radio(
    "Select encoding (check your dataset):",
    ["YES=2, NO=1"],
    index=0
)

if encoding_type == "YES=2, NO=1":
    binary = lambda x: 2 if x == "Yes" else 1

# Create input
input_dict = {}
input_values = [
    binary(smoking), binary(yellow_fingers), binary(anxiety),
    binary(peer_pressure), binary(chronic_disease), binary(fatigue),
    binary(allergy), binary(wheezing), binary(alcohol),
    binary(coughing), binary(shortness_breath), binary(swallowing),
    binary(chest_pain)
]

# Map to exact column names from dataset
for col, val in zip(feature_columns, input_values):
    input_dict[col] = val

input_df = pd.DataFrame([input_dict])

# Show input
with st.expander("View Input Data"):
    st.dataframe(input_df)
    st.write(f"Shape: {input_df.shape}")
    st.write(f"Columns: {input_df.columns.tolist()}")

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict", type="primary"):
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        st.write("---")
        st.subheader("Results")
        
        # Show raw prediction
        st.write(f"**Raw prediction value:** `{prediction}`")
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            st.write(f"**Probabilities:** {proba}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Class 0 (No Cancer)", f"{proba[0]:.1%}")
            with col_b:
                st.metric("Class 1 (Cancer)", f"{proba[1]:.1%}")
        
        # Interpret prediction
        # Adjust this based on your encoding
        cancer_values = [2, 'YES', 'yes', 1]  # Possible values that mean cancer
        
        if prediction in cancer_values:
            st.error("### ⚠️ HIGH RISK: Lung Cancer Detected")
        else:
            st.success("### ✅ LOW RISK: No Lung Cancer Detected")
        
        st.info("⚠️ This is for educational purposes only. Consult a healthcare professional.")
        
        # Test with actual data sample
        st.write("---")
        st.subheader("Model Validation")
        
        # Test with a positive case
        positive_samples = data[data['LUNG_CANCER'].isin(['YES', 'yes', 2, 1])]
        if len(positive_samples) > 0:
            sample = positive_samples.iloc[0]
            sample_features = sample[feature_columns].values.reshape(1, -1)
            sample_pred = model.predict(sample_features)[0]
            st.write(f"✓ Test on known cancer case: Prediction = `{sample_pred}`")
        
         # Test with a negative case
        negative_samples = data[data['LUNG_CANCER'].isin(['NO', 'no', 0])]
        if len(negative_samples) > 0:
            sample_no = negative_samples.iloc[0]
            sample_features_no = sample_no[feature_columns].values.reshape(1, -1)
            sample_pred_no = model.predict(sample_features_no)[0]
            st.write(f"✓ Test on known no-cancer case: Prediction = `{sample_pred_no}`") 
            
       
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
