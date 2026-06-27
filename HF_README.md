---
title: Lung Cancer Prediction API
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# 🫁 Lung Cancer Prediction API

A production-ready **FastAPI** REST API for predicting lung cancer risk using Logistic Regression (91.9% cross-validation accuracy).

**Author:** Mercy Njoki · [GitHub: hug627](https://github.com/hug627)  
**Streamlit App:** https://lungcancer-azansqhe8kw3peeghkink4.streamlit.app/

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch prediction (max 100) |

## 📬 Example Request

```bash
curl -X POST "https://hug627-lung-cancer-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "GENDER": 1, "AGE": 45, "SMOKING": 2, "YELLOW_FINGERS": 2,
    "ANXIETY": 1, "PEER_PRESSURE": 1, "CHRONIC_DISEASE": 1,
    "FATIGUE": 2, "ALLERGY": 1, "WHEEZING": 2,
    "ALCOHOL_CONSUMING": 1, "COUGHING": 2,
    "SHORTNESS_OF_BREATH": 2, "SWALLOWING_DIFFICULTY": 1,
    "CHEST_PAIN": 2
  }'
```

## 🌐 Interactive Docs

Visit `/docs` for full Swagger UI.

## ⚠️ Disclaimer

For educational/research purposes only. Always consult a qualified medical professional.
