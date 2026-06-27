from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import joblib
import numpy as np
import os
from datetime import datetime

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
app = FastAPI(
    title="Lung Cancer Prediction API",
    description="Logistic Regression model predicting lung cancer risk from clinical symptoms. Built by Mercy Njoki.",
    version="1.0.0",
    contact={
        "name": "Mercy Njoki",
        "url": "https://github.com/hug627",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load model (once at startup)
# ─────────────────────────────────────────
# Looks for cancer.pkl in the same folder (root of your repo), or override via env var
MODEL_PATH = os.getenv("MODEL_PATH", "cancer.pkl")

@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. "
            "Make sure cancer.pkl is in the same folder as main.py, "
            "or set the MODEL_PATH environment variable."
        )
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")


# ─────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────
class PatientFeatures(BaseModel):
    """
    13 clinical features used by the Logistic Regression model.
    Binary features: 1 = No, 2 = Yes  (as per the original dataset encoding)
    Age: integer years
    """
    GENDER: int = Field(..., ge=0, le=1, description="Gender: 0 = Female, 1 = Male")
    AGE: int = Field(..., ge=1, le=120, description="Age in years")
    SMOKING: int = Field(..., ge=1, le=2, description="Smoking: 1=No, 2=Yes")
    YELLOW_FINGERS: int = Field(..., ge=1, le=2, description="Yellow fingers: 1=No, 2=Yes")
    ANXIETY: int = Field(..., ge=1, le=2, description="Anxiety: 1=No, 2=Yes")
    PEER_PRESSURE: int = Field(..., ge=1, le=2, description="Peer pressure: 1=No, 2=Yes")
    CHRONIC_DISEASE: int = Field(..., ge=1, le=2, description="Chronic disease: 1=No, 2=Yes")
    FATIGUE: int = Field(..., ge=1, le=2, description="Fatigue: 1=No, 2=Yes")
    ALLERGY: int = Field(..., ge=1, le=2, description="Allergy: 1=No, 2=Yes")
    WHEEZING: int = Field(..., ge=1, le=2, description="Wheezing: 1=No, 2=Yes")
    ALCOHOL_CONSUMING: int = Field(..., ge=1, le=2, description="Alcohol consumption: 1=No, 2=Yes")
    COUGHING: int = Field(..., ge=1, le=2, description="Coughing: 1=No, 2=Yes")
    SHORTNESS_OF_BREATH: int = Field(..., ge=1, le=2, description="Shortness of breath: 1=No, 2=Yes")
    SWALLOWING_DIFFICULTY: int = Field(..., ge=1, le=2, description="Swallowing difficulty: 1=No, 2=Yes")
    CHEST_PAIN: int = Field(..., ge=1, le=2, description="Chest pain: 1=No, 2=Yes")

    class Config:
        schema_extra = {
            "example": {
                "GENDER": 1,
                "AGE": 45,
                "SMOKING": 2,
                "YELLOW_FINGERS": 2,
                "ANXIETY": 1,
                "PEER_PRESSURE": 1,
                "CHRONIC_DISEASE": 1,
                "FATIGUE": 2,
                "ALLERGY": 1,
                "WHEEZING": 2,
                "ALCOHOL_CONSUMING": 1,
                "COUGHING": 2,
                "SHORTNESS_OF_BREATH": 2,
                "SWALLOWING_DIFFICULTY": 1,
                "CHEST_PAIN": 2,
            }
        }


class PredictionResponse(BaseModel):
    prediction: str               # "HIGH RISK" | "LOW RISK"
    prediction_code: int          # 1 | 0
    probability_high_risk: float  # 0.0 – 1.0
    probability_low_risk: float
    model_confidence: str         # "High" | "Moderate" | "Low"
    timestamp: str
    disclaimer: str


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
FEATURE_ORDER = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
    "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
]

def confidence_label(prob: float) -> str:
    if prob >= 0.80 or prob <= 0.20:
        return "High"
    elif prob >= 0.65 or prob <= 0.35:
        return "Moderate"
    return "Low"


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Lung Cancer Prediction API is live 🫁",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", tags=["Health"])
def health():
    """Check if the API and model are running correctly."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the trained model."""
    return {
        "model_type": "Logistic Regression",
        "framework": "scikit-learn",
        "features": FEATURE_ORDER,
        "num_features": len(FEATURE_ORDER),
        "cross_validation_accuracy": "91.9%",
        "target": "LUNG_CANCER (YES / NO)",
        "encoding": {
            "binary_features": "1 = No, 2 = Yes",
            "GENDER": "0 = Female, 1 = Male",
        },
        "author": "Mercy Njoki",
        "model_file": "cancer.pkl",
        "github": "https://github.com/hug627/lung_cancer",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientFeatures):
    """
    Predict lung cancer risk for a patient.

    Returns risk label, probabilities, and confidence level.
    """
    try:
        features = np.array([[getattr(patient, f) for f in FEATURE_ORDER]])
        pred_code = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]  # [prob_NO, prob_YES]

        # Map model output — adjust indices if your label encoding differs
        prob_yes = float(proba[1])  # probability of LUNG_CANCER = YES
        prob_no = float(proba[0])

        label = "HIGH RISK" if pred_code == 1 else "LOW RISK"

        return PredictionResponse(
            prediction=label,
            prediction_code=pred_code,
            probability_high_risk=round(prob_yes, 4),
            probability_low_risk=round(prob_no, 4),
            model_confidence=confidence_label(prob_yes),
            timestamp=datetime.utcnow().isoformat(),
            disclaimer=(
                "This prediction is for educational/research purposes only. "
                "Always consult a qualified medical professional for diagnosis."
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: list[PatientFeatures]):
    """Predict for multiple patients at once (max 100)."""
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100 patients.")
    results = []
    for p in patients:
        features = np.array([[getattr(p, f) for f in FEATURE_ORDER]])
        pred_code = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        results.append({
            "prediction": "HIGH RISK" if pred_code == 1 else "LOW RISK",
            "probability_high_risk": round(float(proba[1]), 4),
        })
    return {"count": len(results), "predictions": results}
