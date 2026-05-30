import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import HopSkipJump

print("=== INITIALIZING AUTOMATED VULNERABILITY SCANNING (PHASE VII) ===")

# 1. Load Assets
model = xgb.XGBClassifier()
model.load_model("security/credit_model.json")
fair_model = joblib.load("security/fair_credit_model.pkl")
expected_features = model.get_booster().feature_names

df = pd.read_csv('data/silver/test_set.csv')

# Isolate ALL Denied records for a batch sweep (limiting to a sample subset for performance evaluation)
df_denied_pool = df[df['target'] == 0].copy()
df_approved_pool = df[df['target'] == 1].copy()

# Sample 10 random rows to run a batch benchmark scan
scan_samples = df_denied_pool.sample(n=10, random_state=101)
anchor_samples = df_approved_pool.sample(n=10, random_state=101)

# 2. Calibrate ART Wrapper
art_classifier = XGBoostClassifier(
    model=model, 
    nb_features=len(expected_features), 
    nb_classes=2
)

attack = HopSkipJump(
    classifier=art_classifier, 
    targeted=True, 
    max_iter=15, # Balanced for batch performance
    max_eval=1000, 
    init_eval=200,
    verbose=False # Turn off logging inside loop
)

# 3. Running the Batch Sweep
results_log = []

print(f"Starting batch simulation across {len(scan_samples)} high-risk profiles...")

for i in range(len(scan_samples)):
    # Extract specific target record and its corresponding anchor
    x_target = scan_samples[expected_features].iloc[i:i+1].values.astype(np.float32)
    x_anchor = anchor_samples[expected_features].iloc[i:i+1].values.astype(np.float32)
    
    # Run the evasion attempt
    y_target = np.array([1], dtype=int)
    x_adv = attack.generate(x=x_target, y=y_target, x_adv_init=x_anchor)
    
    # Evaluate Base Model Evasion
    base_pred = model.predict(x_adv)[0]
    
    # Evaluate Firewall Mitigation
    x_adv_df = pd.DataFrame(x_adv, columns=expected_features)
    SENSITIVE_COL = 'Attribute9_A92'
    sensitive_series = anchor_samples['Attribute9_A92'].iloc[i:i+1].copy()
    sensitive_series.index = x_adv_df.index
    
    firewall_pred = fair_model.predict(x_adv_df, sensitive_features=sensitive_series)[0]
    
    # Categorize Security Outcome
    if base_pred == 1 and firewall_pred == 0:
        outcome = "Mitigated by Firewall"
    elif base_pred == 1 and firewall_pred == 1:
        outcome = "System Breach"
    else:
        outcome = "Evasion Failed"
        
    results_log.append({
        'Sample_ID': i,
        'Base_Model_Exploited': int(base_pred == 1),
        'Firewall_Defended': int(base_pred == 1 and firewall_pred == 0),
        'Security_Outcome': outcome
    })
    print(f" -> Profile {i+1}/{len(scan_samples)} Complete: {outcome}")

# 4. Generate Security Coverage Summary
summary_df = pd.DataFrame(results_log)
total_exploits = summary_df['Base_Model_Exploited'].sum()
total_defended = summary_df['Firewall_Defended'].sum()

print("\n=== BATCH SECURITY METRICS SUMMARY ===")
print(f"Total Profiles Scanned: {len(summary_df)}")
print(f"Base Model Exploit Rate: {(total_exploits / len(summary_df)) * 100:.2f}%")
if total_exploits > 0:
    print(f"Firewall Mitigation Efficiency: {(total_defended / total_exploits) * 100:.2f}%")
else:
    print("Firewall Mitigation Efficiency: N/A (No base model exploits detected)")