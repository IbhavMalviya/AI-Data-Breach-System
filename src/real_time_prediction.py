import pandas as pd
import time
import joblib
import os
from simulate_stream import stream_data

# Load model, encoders, and feature columns
model_dir = os.path.join(os.path.dirname(__file__),'..', 'Model')
model = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))

# Your preprocessing function
from predict import preprocess_new_data_sample  # this must already exist

# Run live simulation
for chunk in stream_data("./Data/UNSW-NB15_1.csv", chunk_size=1, delay=2):
    processed = preprocess_new_data_sample(chunk, encoders, feature_columns)
    prediction = model.predict(processed)[0]
    label = "🔴 Malicious" if prediction == 1 else "🟢 Normal"
    print(f"{label} traffic detected → Row: {chunk.index[0]}")
