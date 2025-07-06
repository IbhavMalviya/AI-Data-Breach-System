# simulate_stream.py
import pandas as pd
import time

def stream_data(csv_path, chunk_size=1, delay=1):
    df = pd.read_csv(csv_path)
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]
        time.sleep(delay)
