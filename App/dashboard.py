import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
st.set_page_config(page_title="AI Data Breach Detector", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2, h3 {color: #222222;}
    .stButton>button {margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/48/lock.png", width=40)
st.sidebar.title("Data Breach Detector 🔐")
st.sidebar.markdown("Built with ❤️ using Machine Learning")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed by **Ibhav Malviya**")
st.sidebar.markdown("""
<div style="text-align: center;">
    <a href="https://www.linkedin.com/in/ibhavmalviya" target="_blank" style="text-decoration: none;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="24" style="vertical-align: middle; margin-right: 6px;"/>
        <span style="font-size: 15px; color: #0077B5;">LinkedIn</span>
    </a><br><br>
    <a href="https://github.com/IbhavMalviya" target="_blank" style="text-decoration: none;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="24" style="vertical-align: middle; margin-right: 6px;"/>
        <span style="font-size: 15px; color: #000000;">GitHub</span>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("🤖 Choose a model", ["Random Forest", "XGBoost", "MLPClassifier"])

# --- Load Models and Preprocessors ---
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, '..', 'Model')
model_paths = {
    "Random Forest": os.path.join(model_dir, 'random_forest.pkl'),
    "XGBoost": os.path.join(model_dir, 'xgboost.pkl'),
    "MLPClassifier": os.path.join(model_dir, 'mlp.pkl'),
}

encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))

categorical_cols = ['proto', 'state', 'service']
drop_cols = ['srcip', 'sport', 'dstip', 'dsport', 'attack_cat', 'label']

# --- Preprocessing Function ---
def preprocess_input(df):
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].map(lambda val: le.transform([val])[0] if val in le.classes_ else -1)

    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    if 'ct_ftp_cmd' in df.columns:
        df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'].replace(' ', pd.NA), errors='coerce')
    df = df.fillna(df.median(numeric_only=True))

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

# --- Tabs UI ---
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Predict", "📊 Visualize", "📈 SHAP Explainability", "🏁 Model Comparison"])

with tab1:
    st.header("📁 Upload CSV to Detect Data Breaches")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)
        
        model = joblib.load(model_paths[selected_model_name])
        processed = preprocess_input(df)
        probs = model.predict_proba(processed)
        preds = model.predict(processed)

        results = df.copy()
        results['Prediction'] = ['Malicious' if p == 1 else 'Normal' for p in preds]
        results['Prob_Normal'] = probs[:, 0]
        results['Prob_Malicious'] = probs[:, 1]

        st.success("✅ Prediction Completed")
        st.dataframe(results[['Prediction', 'Prob_Normal', 'Prob_Malicious']], use_container_width=True)

        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")

        st.session_state.results = results  # Save to session for reuse in other tabs

with tab2:
    st.header("📊 Prediction Visualizations")
    if "results" in st.session_state:
        results = st.session_state.results

        pred_counts = results['Prediction'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=500)
        sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="Set2", ax=ax1)
        ax1.set_title("Prediction Counts")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 3), dpi=500)
        sns.histplot(results['Prob_Malicious'], bins=20, color="crimson", kde=True, label="Malicious")
        sns.histplot(results['Prob_Normal'], bins=20, color="green", kde=True, label="Normal")
        ax2.set_title("Probability Distribution")
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(3, 3), dpi=500)
        ax3.pie(pred_counts, labels=pred_counts.index, autopct="%1.1f%%", colors=["green", "crimson"])
        ax3.set_title("Prediction Breakdown")
        st.pyplot(fig3)
    else:
        st.info("Upload a file first to visualize predictions.")

with tab3:
    st.header("📈 SHAP Feature Explainability")
    shap_path = os.path.join(model_dir, 'rf_shap_summary.png')
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary Plot for Random Forest", use_container_width=True)
    else:
        st.warning("SHAP plot not found in Model folder.")

with tab4:
    st.header("🏁 Compare Model Performance")
    metrics_path = os.path.join(model_dir, 'model_metrics.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.warning("model_metrics.csv not found.")
