#  Import necessary libraries
import pandas as pd
import numpy as np
import time
import joblib
import os
import matplotlib.pyplot as plt
import shap

# Importing sklearn and other libraries for model training and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Loading the data (Same as usual)
columns = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
    'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

df1 = pd.read_csv('./Data/UNSW-NB15_1.csv', names=columns, skiprows=1, low_memory=False)
df2 = pd.read_csv('./Data/UNSW-NB15_2.csv', names=columns, skiprows=1, low_memory=False)
df = pd.concat([df1, df2], ignore_index=True)
df.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'attack_cat'], inplace=True)

# Preprocessing the data again
cat_columns = ['proto', 'service', 'state']
label_encoders = {}
for col in cat_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'].replace(' ', np.nan), errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=['label'])
y = df['label']

# Splitting and applying SMOTE over the dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scaling the data using StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training & Evaluation of 3 different models
def train_and_evaluate(model, name):
    print(f"\n🔧 Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📌 {name} Evaluation")
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Time: {end - start:.2f}s")
    print("Confusion Matrix:")
    print(cm)

    return name, acc, f1, end - start

results = []
# Training and evaluating Random Forest
print("\n🔧 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
start = time.time()
rf.fit(X_train, y_train)
end = time.time()

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📌 Random Forest Evaluation")
print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Time: {end - start:.2f}s")
print("Confusion Matrix:")
print(cm)

results.append(('Random Forest', acc, f1, end - start))

# Training and evaluating XGBoost
print("\n🔧 Training XGBoost...")
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
start = time.time()
xgb.fit(X_train, y_train)
end = time.time()

y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📌 XGBoost Evaluation")
print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Time: {end - start:.2f}s")
print("Confusion Matrix:")
print(cm)

results.append(('XGBoost', acc, f1, end - start))

# Training and evaluating MLPClassifier (Neural Network)
print("\n🔧 Training MLPClassifier (Neural Net)...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)

start = time.time()
mlp.fit(X_train_scaled, y_train)
end = time.time()

y_pred = mlp.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📌 MLPClassifier Evaluation")
print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Time: {end - start:.2f}s")
print("Confusion Matrix:")
print(cm)

results.append(('MLPClassifier (Neural Net)', acc, f1, end - start))


# Comparison Table of all the models
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score', 'Train Time (s)'])
print("\n\n📊 Model Comparison:")
print(results_df.to_string(index=False))


# Plotting Accuracy & F1 Score of each model to compare them visually
model_names = [r[0] for r in results]
accuracies = [r[1] for r in results]
f1_scores = [r[2] for r in results]

plt.figure(figsize=(10, 4), dpi=120)

# Accuracy plot
plt.subplot(1, 2, 1)
plt.bar(model_names, accuracies, color='skyblue')
plt.title('Model Accuracy')
plt.ylim(0.9, 1.0)
plt.ylabel('Accuracy')

# F1 Score plot
plt.subplot(1, 2, 2)
plt.bar(model_names, f1_scores, color='salmon')
plt.title('Model F1 Score')
plt.ylim(0.8, 1.0)
plt.ylabel('F1 Score')

plt.tight_layout()
plt.show()

# SHAP Explainability
explainer = shap.Explainer(xgb, X_test[:100])
shap_values = explainer(X_test[:100])

# SHAP Summary Plot
if isinstance(shap_values, list) and len(shap_values) == 2:
    # For binary classification, SHAP gives one array per class
    shap.summary_plot(shap_values[1], X_test[:100], show=False)
else:
    shap.summary_plot(shap_values, X_test[:100], show=False)
import matplotlib.pyplot as plt
plt.savefig("Model/rf_shap_summary.png", dpi=300, bbox_inches='tight')


# Definining model directory
model_dir = os.path.join(os.path.dirname(__file__), '..', 'Model')
os.makedirs(model_dir, exist_ok=True)

# Saving models
joblib.dump(rf, os.path.join(model_dir, 'random_forest.pkl'))
joblib.dump(xgb, os.path.join(model_dir, 'xgboost.pkl'))
joblib.dump(mlp, os.path.join(model_dir, 'mlp.pkl'))

# Saving results as CSV
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Train Time (s)"])
results_df.to_csv(os.path.join(model_dir, 'model_metrics.csv'), index=False)

print("✅ Models and evaluation metrics saved to /Model folder.")
