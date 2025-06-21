# AI-Powered Data Breach Detection System 🔐

A machine learning-based system that analyzes network traffic to detect early signs of data breaches. Trained on the UNSW-NB15 dataset, it classifies network activity as normal or malicious and is designed to support real-time prediction and alert generation in future versions.

---

## 🚀 Project Status

- 🟢 **Started:** June 2025  
- ✅ Dataset loaded and validated  
- ✅ Data preprocessing pipeline completed  
- ✅ Random Forest model trained and saved  
- 🛠️ **Next up:** Prediction pipeline + deployment preparation  

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
├── Notebooks/ # Jupyter notebooks for EDA and experimentation
├── src/ # Source code
│ ├── data_processing.py # Data loading and preprocessing functions
│ ├── model.py # Training, evaluation, and model saving
│ └── predict.py # [Upcoming] Prediction script for new data
├── main.py # [Optional] Project entry script
├── requirements.txt # Project dependencies
└── README.md # Project overview


---

## 📈 Progress Log

| Date       | Task Completed                                  |
|------------|--------------------------------------------------|
| 2025-06-21 | ✅ GitHub project initialized                    |
| 2025-06-21 | ✅ Folder structure and starter files created    |
| 2025-06-21 | ✅ Dataset loaded and verified                   |
| 2025-06-21 | ✅ Data preprocessing completed using `sklearn` |
| 2025-06-21 | ✅ Model trained and evaluated (Random Forest)   |
| 2025-06-21 | ✅ Model saved using `joblib`                    |
| 2025-06-21 | ✅ Pushed to GitHub (Note: model uses LFS due to size) |

---

## 🧠 Model Overview

- **Algorithm:** `RandomForestClassifier` with `class_weight='balanced'`
- **Accuracy:** ~99.8% on test set  
- **Precision (malicious):** 0.98  
- **Recall (malicious):** 0.97  
- **Key Fix:** `ct_ftp_cmd` column had mixed string/NaN values → cleaned using `pd.to_numeric` + median fill  

---

## 🧰 Tech Stack

- Python 3.11  
- Pandas, NumPy  
- Scikit-learn  
- Joblib (for saving models)  
- Git & GitHub (with LFS for large model files)

---

## 🔮 Next Steps

- [ ] Implement `predict.py` for loading and predicting on new samples  
- [ ] Save and load label encoders for consistent transformation  
- [ ] Add robust logging and exception handling  
- [ ] Explore model optimization or alternati

## 🙋‍♂️ Author

**Ibhav Malviya**  
LinkedIn: https://www.linkedin.com/in/ibhavmalviya
GitHub: https://github.com/IbhavMalviya