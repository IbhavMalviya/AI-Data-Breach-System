import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define values for categorical columns
proto_options = ['tcp', 'udp', 'icmp']
state_options = ['FIN', 'CON', 'INT', 'REQ', 'RST', 'ACC']
service_options = ['http', 'ftp', 'dns', 'ssh', '-', 'smtp']

# Number of rows to generate
n_rows = 25023

# Generate synthetic data
data = {
    'srcip': ['{}.{}.{}.{}'.format(*np.random.randint(0, 255, 4)) for _ in range(n_rows)],
    'sport': np.random.randint(1024, 65535, n_rows),
    'dstip': ['{}.{}.{}.{}'.format(*np.random.randint(0, 255, 4)) for _ in range(n_rows)],
    'dsport': np.random.choice([80, 443, 21, 22, 53], n_rows),
    'proto': np.random.choice(proto_options, n_rows),
    'state': np.random.choice(state_options, n_rows),
    'dur': np.random.exponential(scale=1.0, size=n_rows),
    'sbytes': np.random.randint(0, 100000, n_rows),
    'dbytes': np.random.randint(0, 100000, n_rows),
    'sttl': np.random.choice([64, 128, 255], n_rows),
    'dttl': np.random.choice([64, 128, 255], n_rows),
    'sloss': np.random.randint(0, 5, n_rows),
    'dloss': np.random.randint(0, 5, n_rows),
    'service': np.random.choice(service_options, n_rows),
    'Sload': np.random.rand(n_rows) * 100,
    'Dload': np.random.rand(n_rows) * 100,
    'Spkts': np.random.randint(1, 100, n_rows),
    'Dpkts': np.random.randint(1, 100, n_rows),
    'swin': np.random.randint(1000, 30000, n_rows),
    'dwin': np.random.randint(1000, 30000, n_rows),
    'stcpb': np.random.randint(0, 1e6, n_rows),
    'dtcpb': np.random.randint(0, 1e6, n_rows),
    'smeansz': np.random.randint(40, 1500, n_rows),
    'dmeansz': np.random.randint(40, 1500, n_rows),
    'trans_depth': np.random.randint(0, 10, n_rows),
    'res_bdy_len': np.random.randint(0, 5000, n_rows),
    'Sjit': np.random.rand(n_rows),
    'Djit': np.random.rand(n_rows),
    'Stime': np.random.rand(n_rows) * 1000,
    'Ltime': np.random.rand(n_rows) * 1000,
    'Sintpkt': np.random.rand(n_rows),
    'Dintpkt': np.random.rand(n_rows),
    'tcprtt': np.random.rand(n_rows),
    'synack': np.random.rand(n_rows),
    'ackdat': np.random.rand(n_rows),
    'is_sm_ips_ports': np.random.randint(0, 2, n_rows),
    'ct_state_ttl': np.random.randint(0, 10, n_rows),
    'ct_flw_http_mthd': np.random.randint(0, 5, n_rows),
    'is_ftp_login': np.random.randint(0, 2, n_rows),
    'ct_ftp_cmd': np.random.randint(0, 10, n_rows),
    'ct_srv_src': np.random.randint(0, 20, n_rows),
    'ct_srv_dst': np.random.randint(0, 20, n_rows),
    'ct_dst_ltm': np.random.randint(0, 20, n_rows),
    'ct_src_ltm': np.random.randint(0, 20, n_rows),
    'ct_src_dport_ltm': np.random.randint(0, 20, n_rows),
    'ct_dst_sport_ltm': np.random.randint(0, 20, n_rows),
    'ct_dst_src_ltm': np.random.randint(0, 20, n_rows),
    'attack_cat': np.random.choice(['Normal', 'DoS', 'Reconnaissance', 'Exploits'], n_rows),
    'label': np.random.randint(0, 2, n_rows)
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("test_model_input.csv", index=False)
print("✅ Generated synthetic test dataset: test_model_input.csv")
