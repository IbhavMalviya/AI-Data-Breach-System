import pandas as pd
import random
import joblib

# Load encoders and feature columns
label_encoders = joblib.load("Model/label_encoders.pkl")
feature_columns = joblib.load("Model/feature_columns.pkl")

# Function to safely choose a valid encoded value from encoder
def get_valid_value(col):
    if col in label_encoders:
        return random.choice(label_encoders[col].classes_)
    return 'unknown'

# Generate synthetic malicious sample
def generate_malicious_sample():
    return {
        'srcip': '10.0.0.1',
        'sport': random.randint(1024, 65535),
        'dstip': '192.168.1.1',
        'dsport': random.choice([21, 22, 23, 80, 443, 8080]),
        'proto': get_valid_value('proto'),
        'state': get_valid_value('state'),
        'dur': round(random.uniform(0.01, 1.5), 3),
        'sbytes': random.randint(50000, 1000000),
        'dbytes': random.randint(20000, 800000),
        'sttl': random.randint(20, 255),
        'dttl': random.randint(20, 255),
        'sloss': random.randint(0, 5),
        'dloss': random.randint(0, 5),
        'service': get_valid_value('service'),
        'Sload': round(random.uniform(0, 100000), 2),
        'Dload': round(random.uniform(0, 50000), 2),
        'Spkts': random.randint(10, 1000),
        'Dpkts': random.randint(10, 1000),
        'swin': random.randint(1000, 65535),
        'dwin': random.randint(1000, 65535),
        'stcpb': random.randint(0, 1000000),
        'dtcpb': random.randint(0, 1000000),
        'smeansz': random.randint(50, 1500),
        'dmeansz': random.randint(50, 1500),
        'trans_depth': random.randint(0, 10),
        'res_bdy_len': random.randint(0, 5000),
        'Sjit': round(random.uniform(0.0, 5.0), 3),
        'Djit': round(random.uniform(0.0, 5.0), 3),
        'Stime': round(random.uniform(0.0, 1.0), 3),
        'Ltime': round(random.uniform(0.0, 1.0), 3),
        'Sintpkt': round(random.uniform(0.001, 1.0), 3),
        'Dintpkt': round(random.uniform(0.001, 1.0), 3),
        'tcprtt': round(random.uniform(0.01, 2.0), 3),
        'synack': round(random.uniform(0.01, 1.0), 3),
        'ackdat': round(random.uniform(0.01, 1.0), 3),
        'is_sm_ips_ports': random.randint(0, 1),
        'ct_state_ttl': random.randint(0, 10),
        'ct_flw_http_mthd': random.randint(0, 1),
        'is_ftp_login': random.randint(0, 1),
        'ct_ftp_cmd': random.randint(0, 5),
        'ct_srv_src': random.randint(0, 50),
        'ct_srv_dst': random.randint(0, 50),
        'ct_dst_ltm': random.randint(0, 50),
        'ct_src_ltm': random.randint(0, 50),
        'ct_src_dport_ltm': random.randint(0, 50),
        'ct_dst_sport_ltm': random.randint(0, 50),
        'ct_dst_src_ltm': random.randint(0, 50),
        'attack_cat': 'Malicious',
        'label': 1
    }

# Generate malicious samples
malicious_data = [generate_malicious_sample() for _ in range(50)]
malicious_df = pd.DataFrame(malicious_data)

# Save to CSV
malicious_df.to_csv("malicious_test_cases.csv", index=False)
print("✅ Malicious test data saved to 'malicious_test_cases.csv'")
