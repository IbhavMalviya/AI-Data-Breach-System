🔐 AI-Powered Data Breach Detection System
A production-ready machine learning system to detect early signs of data breaches by analyzing real-time network traffic. Trained on the UNSW-NB15 dataset, this system classifies traffic as normal or malicious with high accuracy and is equipped with a professional Streamlit dashboard, SHAP-based explainability, Docker support, and multi-model evaluation.

------------------------------------------------------------------------------------
📌 Table of Contents

* 🎯 Objectives
* 📁 Project Structure
* 📊 Dataset Used
* 🧠 Model Overview
* 📊 Model Comparison
* 🧪 SHAP Explainability
* 🖥️ Streamlit Dashboard
* 🐳 Docker Support
* 💻 Local Installation
* 🗂️ Notebooks
* 🙋‍♂️ Author
------------------------------------------------------------------------------------

🎯 Objectives
-✅ Detect abnormal/malicious network behavior via ML

-✅ Train, compare and evaluate multiple models

-✅ Provide explainability with SHAP values

-✅ Build a clean, intuitive Streamlit dashboard for live inference

-✅ Enable deployment via both Streamlit and Docker

------------------------------------------------------------------------------------
📁 Project Structure

AI-Data-Breach-System/
│
├── App/
│   └── dashboard.py              # Streamlit App
│
├── Data/
│   ├── UNSW-NB15_1.csv
│   └── UNSW-NB15_2.csv
│
├── Model/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── mlp.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.pkl
│   └── model_metrics.csv
│
├── Notebooks/
│   ├── 1_data_eda.ipynb
│   ├── 2_data_eda.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── predict.py
│   └── train_compare_models.py
│
├── Dockerfile
├── requirements.txt
└── README.md

------------------------------------------------------------------------------------
📊 Dataset Used: UNSW-NB15
- Includes real attack traffic and normal traffic across multiple sessions

- Features like protocol, packet size, duration, services, flow behavior, and more
 
- 49 total features + label (0 = Normal, 1 = Malicious)

- Size: ~2.5M rows (used first 2 subsets for training)
------------------------------------------------------------------------------------

🧠 Model Overview 
| Metric        | Value                                 |
| ------------- | ------------------------------------- |
| Algorithm     | Random Forest                         |
| Accuracy      | 99.8%                                 |
| F1 Score      | 97%                                   |
| Class Weight  | Balanced                              |
| Preprocessing | Label Encoding + SMOTE + NaN Handling |
| Saved Using   | joblib                                |

Confusion Matrix:
[[264542    366]
 [   408  14684]]

Classification Report:
              precision    recall  f1-score   support
     Normal       1.00       1.00      1.00    264908
  Malicious       0.98       0.97      0.97     15092

------------------------------------------------------------------------------------

📊 Model Comparison
| Model                  | Accuracy | F1 Score | Train Time |
| ---------------------- | -------- | -------- | ---------- |
| Random Forest          | 0.9984   | 0.9702   | 1.5s       |
| XGBoost                | 0.9986   | 0.9710   | 6.3s       |
| MLPClassifier (Neural) | 0.9923   | 0.9438   | 23.4s      |


------------------------------------------------------------------------------------
🧪 SHAP Explainability
SHAP (SHapley Additive exPlanations) helps explain individual predictions by attributing contributions to each feature.

- Visualize most influential features causing malicious detection

- Use SHAP summary plot to interpret global model behavior

------------------------------------------------------------------------------------

🖥️ Streamlit Dashboard
Deployed App: https://ai-data-breach-system.streamlit.app
Upload your network log .csv and view predictions + confidence scores instantly!

Key Features:
- ✅ Prediction table with probabilities

- ✅ Pie chart for class distribution

- ✅ Confidence histogram

- ✅ Bar chart of normal vs malicious

- ✅ Downloadable result CSV

- ✅ Professional sidebar with GitHub + LinkedIn

------------------------------------------------------------------------------------

🐳 Docker Support
🧱 Build the Docker image:
docker build -t data-breach-app .

🚀 Run the container:
docker run -p 8501:8501 data-breach-app

Open http://localhost:8501 to use the app.

------------------------------------------------------------------------------------
💻 Local Installation
1. Clone the repository:
git clone https://github.com/IbhavMalviya/AI-Data-Breach-System.git
cd AI-Data-Breach-System

2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run App/dashboard.py

------------------------------------------------------------------------------------

🙋‍♂️ Author
Ibhav Malviya

💼 LinkedIn: linkedin.com/in/ibhavmalviya

💻 GitHub: github.com/IbhavMalviya

If this project helped you or inspired you, consider giving it a ⭐ on GitHub!
