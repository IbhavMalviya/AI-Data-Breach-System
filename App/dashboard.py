import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '..', 'Model', 'random_forest.pkl')
encoders_path = os.path.join(base_dir, '..', 'Model', 'label_encoders.pkl')
features_path = os.path.join(base_dir, '..', 'Model', 'feature_columns.pkl')

model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)
feature_columns = joblib.load(features_path)

categorical_cols = ['proto', 'state', 'service']
drop_cols = ['srcip', 'sport', 'dstip', 'dsport', 'attack_cat', 'label']

def preprocess_input(df, encoders):
    df = df.copy()

    # Encode categorical columns
    for col in categorical_cols:
        if col in df.columns:
            le = encoders[col]
            df[col] = df[col].map(lambda val: le.transform([val])[0] if val in le.classes_ else -1)

    # Drop unused columns
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Fill missing values
    df.fillna(0, inplace=True)

    # Keep only the columns used during training
    df = df[[col for col in feature_columns if col in df.columns]]

    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_columns]
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Data Breach Detector", layout="wide")
st.title("🔐 AI-Powered Data Breach Detection")
st.markdown("Upload your network traffic CSV to classify it as **Normal** or **Malicious**.")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded data preview:")
        st.dataframe(df)

        processed = preprocess_input(df, label_encoders)
        probs = model.predict_proba(processed)
        preds = model.predict(processed)

        results = df.copy()
        results['Prediction'] = ['Malicious' if p == 1 else 'Normal' for p in preds]
        results['Prob_Normal'] = probs[:, 0]
        results['Prob_Malicious'] = probs[:, 1]

        st.success("✅ Prediction completed.")
        st.subheader("📊 Results")
        st.dataframe(results[['Prediction', 'Prob_Normal', 'Prob_Malicious']], use_container_width=True)

                # --- Visualization 1: Bar Chart for Predictions ---
        st.subheader("📊 Prediction Summary")
        pred_counts = results['Prediction'].value_counts()

        fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=500)
        sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="viridis", ax=ax1)
        ax1.set_title("Count of Predictions")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Prediction")
        st.pyplot(fig1)

        # --- Visualization 2: Probability Histogram ---
        st.subheader("🎯 Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 3), dpi=500)
        sns.histplot(results['Prob_Malicious'], bins=20, color="crimson", label="Malicious", kde=True)
        sns.histplot(results['Prob_Normal'], bins=20, color="green", label="Normal", kde=True)
        ax2.set_title("Distribution of Predicted Probabilities")
        ax2.set_xlabel("Probability")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)

        # --- Visualization 3: Pie Chart ---
        st.subheader("🍰 Prediction Ratio")
        fig3, ax3 = plt.subplots(figsize=(3, 3), dpi=500)
        ax3.pie(pred_counts, labels=pred_counts.index, autopct="%1.1f%%", colors=["green", "crimson"])
        ax3.set_title("Prediction Breakdown")
        st.pyplot(fig3)

        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv_out, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction:\n{e}")
else:
    st.info("👆 Upload a CSV file above to begin.")
