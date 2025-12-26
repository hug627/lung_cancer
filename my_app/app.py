import streamlit as st
st.set_page_config(
    page_title="Lung Cancer Prediction Health System",
    page_icon="🫁"
)

st.header("🫁 Lung Cancer Prediction Health System")

st.subheader("📋 Objective of This Project")
st.write("""
Develop a machine learning prediction system that identifies whether a patient has lung cancer 
based on their demographic information, lifestyle factors, and symptoms they are experiencing.

The objective will be achieved through:
""")

st.subheader("1️⃣ Data Understanding")
st.write("""
This project uses the **Survey Lung Cancer Dataset**, which contains patient information including:
- Demographic data (gender, age)
- Lifestyle factors (smoking, alcohol consumption)
- Physical symptoms and health indicators
- Lung cancer diagnosis (target variable)
""")

st.subheader("2️⃣ Main Features Used")
st.write("""
The model uses **15 key features** to predict lung cancer risk:

**Demographic Information:**
1. **Gender** - Patient's biological sex
2. **Age** - Patient's age in years

**Lifestyle Factors:**
3. **Smoking** - Smoking history
4. **Alcohol Consuming** - Alcohol consumption habits
5. **Peer Pressure** - Social influence on health behaviors

**Physical Symptoms:**
6. **Yellow Fingers** - Discoloration of fingers (often smoking-related)
7. **Anxiety** - Presence of anxiety symptoms
8. **Fatigue** - Persistent tiredness or exhaustion
9. **Allergy** - Allergic reactions or sensitivities
10. **Wheezing** - Whistling sound when breathing
11. **Shortness of Breath** - Difficulty breathing
12. **Swallowing Difficulty** - Trouble swallowing (dysphagia)
13. **Chest Pain** - Pain or discomfort in chest area

**Medical History:**
14. **Chronic Disease** - Presence of other chronic conditions

**Target Variable:**
15. **Lung Cancer** - Diagnosis outcome (Yes/No)

These features are analyzed to predict whether a patient has lung cancer.
""")

st.subheader("3️⃣ Understanding the Context")
st.write("""
Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection 
is crucial for improving survival rates and treatment outcomes. However, many cases are diagnosed 
at advanced stages when treatment options are limited.

**The Challenge:**
- Traditional diagnosis methods (CT scans, biopsies) are expensive and invasive
- Many patients don't recognize early warning signs
- Healthcare systems need efficient screening tools for high-risk populations

**Our Solution:**
This machine learning-based prediction system addresses these challenges by:

1. **Early Risk Assessment:** Analyzing patient symptoms and lifestyle factors to identify 
   high-risk individuals who need immediate medical attention

2. **Accessible Screening:** Providing a preliminary assessment tool that can be used before 
   expensive diagnostic procedures

3. **Data-Driven Insights:** Understanding which symptoms and risk factors are most strongly 
   associated with lung cancer diagnosis

4. **Supporting Healthcare Decisions:** Assisting medical professionals in prioritizing patients 
   for further testing and diagnosis

**Project Goals:**

✅ **High Sensitivity (Recall):** Ensure we catch as many cancer cases as possible (minimize false negatives)

✅ **Accurate Predictions:** Provide reliable risk assessments to guide medical decisions

✅ **Interpretable Results:** Help patients and doctors understand which factors contribute to cancer risk

✅ **Accessible Healthcare:** Make preliminary screening more accessible and cost-effective

**Important Note:**
This system is designed as a **screening tool only** and should never replace professional 
medical diagnosis. All positive predictions must be confirmed through proper medical examination, 
imaging, and biopsy by qualified healthcare professionals.
""")

st.subheader("4️⃣ How It Works")
st.write("""
The system uses advanced machine learning algorithms trained on historical patient data to:

1. **Analyze Input Features:** Process patient demographic data, symptoms, and lifestyle factors
2. **Calculate Risk Score:** Generate a probability of lung cancer presence
3. **Provide Prediction:** Classify patients as high-risk or low-risk
4. **Recommend Actions:** Suggest next steps (immediate medical consultation vs. routine monitoring)

**Machine Learning Models Used:**
- Support Vector Machines (SVM)
- Logistic Regression
- Random Forest
- XGBoost
- Neural Networks (MLP)

The models are optimized to maximize cancer detection rate (recall) while maintaining 
reasonable precision to minimize false alarms.
""")

st.subheader("5️⃣ Expected Impact")
st.write("""
By implementing this lung cancer prediction system, we aim to:

🎯 **Improve Early Detection:** Identify high-risk patients before symptoms become severe

💰 **Reduce Healthcare Costs:** Provide cost-effective preliminary screening before expensive tests

⏱️ **Save Time:** Quickly assess large populations and prioritize those needing immediate attention

📊 **Support Research:** Provide insights into lung cancer risk factors and symptom patterns

❤️ **Save Lives:** Enable earlier intervention and treatment, improving patient outcomes

This project demonstrates how machine learning can support public health initiatives and 
assist medical professionals in making data-driven decisions for better patient care.
""")

# Optional: Add a disclaimer section
st.markdown("---")
st.warning("""
⚠️ **Medical Disclaimer:**

This prediction system is for educational and screening purposes only. It does not provide 
medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals 
for medical decisions. If you suspect you have lung cancer or experience concerning symptoms, 
seek immediate medical attention.
""")

st.info("""
ℹ️ **About This Project:**

This is a machine learning project developed to demonstrate the application of AI in healthcare. 
The models are trained on survey data and should be validated with medical professionals before 
any clinical use.
""")