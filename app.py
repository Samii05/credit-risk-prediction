import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Charger modèle et objets ===
model = joblib.load("model/best_adaboost.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
scaler = joblib.load("model/scaler.pkl")
X_columns = joblib.load("model/X_columns.pkl")

# === Constantes ===
cat_columns = joblib.load("model/cat_columns.pkl")
num_columns = joblib.load("model/num_columns.pkl")

# === Titre de l'application ===
st.set_page_config(page_title="Credit Risk App", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.markdown("Remplissez les informations client pour prédire son profil de crédit.")

# === Formulaire utilisateur ===
with st.form("credit_form"):
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", options=["male", "female"])
        job = st.selectbox("Job", options=["0", "1", "2", "3"])
        housing = st.selectbox("Housing", options=["own", "free", "rent"])
        saving_accounts = st.selectbox("Saving accounts", options=["little", "moderate", "quite rich", "rich"])
    with col2:
        checking_account = st.selectbox("Checking account", options=["little", "moderate", "rich", "no checking"])
        purpose = st.selectbox("Purpose", options=[
            "radio/TV", "education", "car", "repairs", "domestic appliances",
            "furniture/equipment", "business", "vacation/others"])
        credit_amount = st.slider("Credit amount", 1, 20000, 1000, step=100)
        duration = st.slider("Duration (months)", 1, 72, 12)
    age = st.slider("Age", 18, 100, 30)

    submit = st.form_submit_button("🔍 Predict")

# === Fonction de prédiction ===
def preprocess_and_predict(input_data):
    df_input = pd.DataFrame([input_data])

    # Appliquer les encodages
    for col in cat_columns:
        le = label_encoders[col]
        try:
            df_input[col] = le.transform(df_input[col])
        except ValueError:
            df_input[col] = -1  # gère les valeurs inconnues

    # Normalisation des colonnes numériques
    df_input[num_columns] = scaler.transform(df_input[num_columns])

    # Réordonner les colonnes
    df_input = df_input.reindex(columns=X_columns, fill_value=0)

    # Prédiction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    return prediction, probability

# === Affichage des résultats ===
if submit:
    input_data = {
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Purpose": purpose,
        "Duration": duration,
        "Credit amount": credit_amount,
        "Age": age
    }

    pred, proba = preprocess_and_predict(input_data)
    if pred == 1:
        st.success(f"✅ Le client est **BON** avec une probabilité de {proba:.2f}")
    else:
        st.error(f"⚠️ Le client est **MAUVAIS** avec une probabilité de {proba:.2f}")
