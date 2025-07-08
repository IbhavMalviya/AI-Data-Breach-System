# Importing necessary libraries
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Paths where our model, encoders, and feature columns are stored
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'Model')
model_path = os.path.join(MODEL_DIR, 'random_forest.pkl')
encoder_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
feature_path = os.path.join(MODEL_DIR, 'feature_columns.pkl')

# Loading model, encoders, and feature columns from the paths
model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
feature_columns = joblib.load(feature_path)

# Defining columns to transform or drop as they are not needed for prediction
# 'proto', 'state', and 'service' are categorical features that will be encoded
categorical_cols = ['proto', 'state', 'service']
drop_cols = ['srcip', 'sport', 'dstip', 'dsport', 'attack_cat', 'label']

def preprocess_new_data_sample(df, label_encoders, feature_columns):
    df = df.copy()

    # Dropping target and unrelated columns
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Encoding the needed categorical features
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].map(lambda val: le.transform([val])[0] if val in le.classes_ else -1)

    # Ensuring that 'ct_ftp_cmd' is numeric and fill any missing values
    if 'ct_ftp_cmd' in df.columns:
        df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'].replace(' ', np.nan), errors='coerce')

    # Filling the remaining NaNs with median
    df = df.fillna(df.median(numeric_only=True))

    # Ensureing all expected columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]  # Ensuring the correct column order

    return df

# Loading input CSV file(New data samples)
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"📁 Loaded file: {csv_path}")
else:
    print("⚠️ No CSV provided. Using a default dummy sample.\n")
    df = pd.DataFrame([{
        'srcip': '1.1.1.1', 'sport': 1234, 'dstip': '2.2.2.2', 'dsport': 80, 'proto': 'tcp', 'state': 'FIN',
        'dur': 0.1, 'sbytes': 1000, 'dbytes': 500, 'sttl': 128, 'dttl': 64, 'sloss': 0, 'dloss': 0,
        'service': 'http', 'Sload': 0.5, 'Dload': 0.1, 'Spkts': 10, 'Dpkts': 5,
        'swin': 2000, 'dwin': 2000, 'stcpb': 0, 'dtcpb': 0, 'smeansz': 100, 'dmeansz': 50,
        'trans_depth': 1, 'res_bdy_len': 0, 'Sjit': 0.0, 'Djit': 0.0, 'Stime': 0.0, 'Ltime': 0.0,            
        'Sintpkt': 0.01, 'Dintpkt': 0.02, 'tcprtt': 0.1, 'synack': 0.01, 'ackdat': 0.01,                             #Default values for dummy sample
        'is_sm_ips_ports': 0, 'ct_state_ttl': 1, 'ct_flw_http_mthd': 0, 'is_ftp_login': 0,
        'ct_ftp_cmd': 0, 'ct_srv_src': 2, 'ct_srv_dst': 3, 'ct_dst_ltm': 2, 'ct_src_ltm': 1,
        'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 2, 'ct_dst_src_ltm': 1,
        'attack_cat': 'Normal', 'label': 0
    }])

# Preprocessing the new data and predicting the results
processed = preprocess_new_data_sample(df, label_encoders, feature_columns)
probs = model.predict_proba(processed)
preds = model.predict(processed)

# Output of the predictions
for i, (p, prob) in enumerate(zip(preds, probs)):
    label = "Malicious" if p == 1 else "Normal"
    print(f"\n🔍 Sample {i+1} prediction: {label}")
    print(f"   → Probability (Normal, Malicious): {prob}")

# Saving the results for our future reference
output_df = df.copy()
output_df['prediction'] = preds
output_df['prob_normal'] = probs[:, 0]
output_df['prob_malicious'] = probs[:, 1]
output_df.to_csv('prediction_output.csv', index=False)
print("\n✅ Prediction completed successfully.")
print("📄 Results saved to 'prediction_output.csv'")


