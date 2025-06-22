import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import load_data, preprocess_dataframe, split_and_clean
import joblib

# Define column names
columns = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
    'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

# Load and preprocess the data
df = load_data(
    "./Data/UNSW-NB15_1.csv",
    "./Data/UNSW-NB15_2.csv",
    columns
)
df, label_encoders = preprocess_dataframe(df)
X_train, X_test, y_train, y_test = split_and_clean(df)

# Step 1: Create the model with class_weight='balanced'
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Step 2: Train the model
rf.fit(X_train, y_train)

# Save feature columns used during training
feature_columns = X_train.columns.tolist()
import os
model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
os.makedirs(model_dir, exist_ok=True)
feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
joblib.dump(feature_columns, feature_columns_path)
print("🧠 Feature columns saved successfully.")


# Step 3: Make predictions
y_pred = rf.predict(X_test)

# Step 4: Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


import os

model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'random_forest.pkl')
joblib.dump(rf, model_path)

def predict_new_data(model, new_data):
    # preprocess new_data same as training
    return model.predict(new_data)

# Save the label encoders to reuse them in prediction
encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
joblib.dump(label_encoders, encoders_path)
print("✅ Model and encoders saved successfully.")
# Save the model
print(f"Model saved to {model_path}")