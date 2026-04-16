import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Load the Test Data (The 'Ground Truth')
test = pd.read_csv('data/silver/test_set.csv')
y_test = test['target']

# 2. Load the Models from the /models directory
model_full = joblib.load('models/baseline_full.pkl')
model_compliant = joblib.load('models/baseline_compliant.pkl')

# 3. Define the features for each model
# Model B (Compliant) needs the protected columns dropped before it can 'predict'
protected_cols = [col for col in test.columns if 'Attribute13' in col or 'Attribute9' in col]
X_test_full = test.drop(columns=['target'])
X_test_compliant = X_test_full.drop(columns=protected_cols)

# 4. Calculate ROC-AUC
for name, model, X in [("Full Model", model_full, X_test_full), 
                       ("Compliant Model", model_compliant, X_test_compliant)]:
    
    # Get the probability of 'Default' (Class 1)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"✅ {name} ROC-AUC: {auc:.4f}")
    