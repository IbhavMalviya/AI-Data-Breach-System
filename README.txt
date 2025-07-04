# AI-Powered Data Breach Detection System 🔐

A machine learning-based system that analyzes network traffic to detect early signs of data breaches. Trained on the UNSW-NB15 dataset, it classifies network activity as normal or malicious and is designed to support real-time prediction and alert generation in future versions.

---

## 🚀 Project Status

- 🟢 **Started:** June 2025  
- ✅ Dataset loaded and validated  
- ✅ Data preprocessing pipeline completed  
- ✅ Random Forest model trained and saved  
- ✅ Prediction pipeline + deployment preparation  

---

## 🎯 Objectives

- Detect abnormal or malicious behavior from network traffic logs  
- Preprocess and clean real-world intrusion detection data (UNSW-NB15)  
- Train and evaluate a machine learning model for binary classification  
- Save trained models for future reuse in predictions  
- Lay the foundation for real-time monitoring and alerting via a dashboard  

---

## 🗂️ Project Structure

AI-Data-Breach-System/
├── Data/ # Raw UNSW-NB15 CSV files
├── Models/ # Trained machine learning models (e.g., random_forest.pkl)
│ ├── random_forest.pkl
│ ├── label_encoders.pkl
│ ├── random_forest.pkl
├── Notebooks/ # Jupyter notebooks for EDA and experimentation
├── src/ # Source code
│ ├── data_processing.py # Data loading and preprocessing functions
│ ├── model.py # Training, evaluation, and model saving
│ └── predict.py # [Upcoming] Prediction script for new data
├── requirements.txt # Project dependencies
└── README.md # Project overview


---

## 📈 Progress Log

| Date       | Task Completed                                  
|------------|--------------------------------------------------
| 2025-06-21 | ✅ GitHub project initialized                    
| 2025-06-21 | ✅ Folder structure and starter files created    
| 2025-06-21 | ✅ Dataset loaded and verified                   
| 2025-06-21 | ✅ Data preprocessing completed using `sklearn` 
| 2025-06-21 | ✅ Model trained and evaluated (Random Forest)   
| 2025-06-21 | ✅ Model saved using `joblib`                    
| 2025-06-21 | ✅ Pushed to GitHub (Note: model uses LFS due to size) 
| 2025-06-22 | ✅ Streamlit frontend added and tested
| 2025-06-22 | ✅ Synthetic data generation & robustness tests
| 2025-06-22 | 🔄 Evaluation on edge cases underway
| 2025-06-22 | ☁️ Deploy to Streamlit Cloud & Hugging Face
---

## 🧠 Model Overview

- **Algorithm:** `RandomForestClassifier` with `class_weight='balanced'`
- **Accuracy:** ~99.8% on test set  
- **Precision (malicious):** 0.98  
- **Recall (malicious):** 0.97  
- **Trained On: Cleaned and processed UNSW-NB15 dataset
- **Key Fix:** `ct_ftp_cmd` column had mixed string/NaN values → cleaned using `pd.to_numeric` + median fill  

Confusion Matrix:
[264542    366]
[408       14684]

Classification Report:
                 precision    recall    f1-score    support

           0       1.00        1.00      1.00       264908
           1       0.98        0.97      0.97        15092

    accuracy                             1.00       280000
   macro avg       0.99        0.99      0.99       280000
weighted avg       1.00        1.00      1.00       280000

---

## 🧰 Tech Stack

- Python 3.11  
- Pandas, NumPy  
- Scikit-learn  
- Joblib (for saving models)  
- Git & GitHub (with LFS for large model files)
- Streamlit (for UI)
---

🖥️ Streamlit Dashboard
Upload a .csv file containing network traffic logs to see instant predictions with visual feedback.

Features:
✅ Tabular prediction output with confidence scores

✅ Pie chart: Prediction distribution

✅ Histogram: Confidence levels

✅ Bar chart: Total counts

✅ Downloadable CSV with predictions


📦 Installation
git clone https://github.com/IbhavMalviya/AI-Data-Breach-System.git
cd AI-Data-Breach-System
pip install -r requirements.txt

Run it locally:
streamlit run app/dashboard.py


## 🙋‍♂️ Author

**Ibhav Malviya**  
LinkedIn: https://www.linkedin.com/in/ibhavmalviya
GitHub: https://github.com/IbhavMalviya
