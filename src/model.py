# Importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import load_data, preprocess_dataframe, split_and_clean
import joblib
import os
import numpy as np
from imblearn.over_sampling import SMOTE

# Defining column names
columns = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
    'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

# Loading and preprocessing the data 
df = load_data(
    "./Data/UNSW-NB15_1.csv",
    "./Data/UNSW-NB15_2.csv",
    columns
)
df, label_encoders = preprocess_dataframe(df)

# Converting to numeric and filling missing values in our dataset
df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'].replace(' ', np.nan), errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

# Splitting the data set into train and test sets
X_train, X_test, y_train, y_test = split_and_clean(df)

# Apply SMOTE to training data to balance the classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#Creating the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

#Training the model
rf.fit(X_train, y_train)

# Saving feature columns used during our model training to Model directory
feature_columns = X_train.columns.tolist()
model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
os.makedirs(model_dir, exist_ok=True)
feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
joblib.dump(feature_columns, feature_columns_path)
print("🧠 Feature columns saved successfully.")

# Step 3: Making predictions based on the trained model with the test set
y_pred = rf.predict(X_test)

# Step 4: Evaluation of the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving model and encoders to model directory
model_path = os.path.join(model_dir, 'random_forest.pkl')
joblib.dump(rf, model_path)
encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
joblib.dump(label_encoders, encoders_path)

print("✅ Model and encoders saved successfully.")
print(f"Model saved to {model_path}")

#Prediction wrapper easier to use the model for new data
def predict_new_data(model, new_data):
    return model.predict(new_data)
