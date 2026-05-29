import numpy as np
import pandas as pd
import xgboost as xgb
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import HopSkipJump

# 1. Load Model
model = xgb.XGBClassifier()
model.load_model("security/credit_model.json")
expected_features = model.get_booster().feature_names

# 2. Load Dataset
df = pd.read_csv('data/silver/test_set.csv')

# 3. Select our "Denied" target (the record we want to illegally flip to Approved)
df_denied = df[df['target'] == 0]
x_denied_raw = df_denied.sample(n=1, random_state=42)
x_test_denied = x_denied_raw[expected_features].values.astype(np.float32)

# 4. NEW: Select an "Approved" helper record to act as the initialization anchor
df_approved = df[df['target'] == 1]
if df_approved.empty:
    raise ValueError("No approved records found in the test set to use as an initialization anchor!")

x_approved_raw = df_approved.sample(n=1, random_state=42) # Pick a valid approved record
x_test_approved = x_approved_raw[expected_features].values.astype(np.float32)

print(f"Denied Target Shape: {x_test_denied.shape} | Approved Anchor Shape: {x_test_approved.shape}")

# 5. Wrap Model
art_classifier = XGBoostClassifier(
    model=model, 
    nb_features=len(expected_features), 
    nb_classes=2
)

# 6. Configure Targeted HopSkipJump
# targeted=True requires the algorithm to march toward a specific class output
attack = HopSkipJump(
    classifier=art_classifier, 
    targeted=True, 
    max_iter=30, 
    max_eval=2000, 
    init_eval=100, # Can be lower now because we aren't guessing randomly
    verbose=True
)

print("\n--- Red Team Engine Calibrated Successfully (Targeted Mode) ---")

# 7. Execute the Attack
# We specify y_target=[1] (Approved) and pass x_adv_init as our starting anchor point
print("Generating adversarial evasion sample using targeted initialization...")
y_target = np.array([1], dtype=int)

x_adv = attack.generate(
    x=x_test_denied, 
    y=y_target, 
    x_adv_init=x_test_approved  # This guides the algorithm directly to the boundary
)

import joblib

# ... (Previous code remains the same through generating x_adv) ...

# 8. Evaluate Base Model Results
original_base_pred = model.predict(x_test_denied)[0]
adversarial_base_pred = model.predict(x_adv)[0]

print("\n=== BASE MODEL EVASION RESULTS ===")
print(f"Original Base Prediction (0=Denied, 1=Approved): {original_base_pred}")
print(f"Adversarial Base Prediction (0=Denied, 1=Approved): {adversarial_base_pred}")

# 9. GOVERNANCE FIREWALL EVALUATION
print("\n=== GOVERNANCE LAYER (FAIRLEARN) FIREWALL SCAN ===")
try:
    # Load the original Fairlearn wrapper we saved earlier
    fair_model = joblib.load("security/fair_credit_model.pkl")
    
    # Reconstruct the DataFrame for Fairlearn with the exact expected features
    x_adv_df = pd.DataFrame(x_adv, columns=expected_features)
    
    # STRATEGIC FIX: Extract the sensitive feature directly from the raw anchor dataframe row
    # Fairlearn needs the exact sensitive series alignment. We mirror the index to match x_adv_df.
    # We will look for the base 'Attribute9' or use the full raw row slice to isolate it.
    
    # Let's find the columns in the original test set that represent the sensitive features
    # (Fairlearn optimizer was trained on a specific column from the silver data)
    possible_sensitive_cols = [col for col in x_approved_raw.columns if 'Attribute9' in col or 'Attribute13' in col]
    
    # Isolate the exact feature slice from our raw approved anchor row
    sensitive_data = x_approved_raw[possible_sensitive_cols].iloc[0:1]
    
    # If Fairlearn expects a single series/column, grab the first one used as the constraint
    # Change 'Attribute9_A92' if your Fairlearn pipeline used a different base attribute
    SENSITIVE_COL = 'Attribute9_A92' 
    sensitive_series = x_approved_raw[SENSITIVE_COL].copy()
    sensitive_series.index = x_adv_df.index # Force index alignment (0)
    
    # Pass the adversarial sample through the Fairness Post-Processing Gate
    governance_pred = fair_model.predict(x_adv_df, sensitive_features=sensitive_series)[0]
    
    print(f"Fairlearn Firewall Decision (0=Denied, 1=Approved): {governance_pred}")
    
    print("\n=== SYSTEM OVERALL SECURITY POSTURE ===")
    if adversarial_base_pred == 1 and governance_pred == 0:
        print("CONCLUSION: DEFENSE SUCCESSFUL! The base model was fooled, but the Fairlearn Governance Firewall caught the variance and enforced a DENIAL.")
    elif adversarial_base_pred == 1 and governance_pred == 1:
        print("CONCLUSION: SYSTEM BREACHED. The attack successfully bypassed both the base model and the fairness threshold.")
    else:
        print("CONCLUSION: No evasion detected at the base level.")

except Exception as e:
    print(f"Governance evaluation failed: {e}")
    print("Diagnostic: Let's see what features Fairlearn is expecting by running: print(fair_model)")
    