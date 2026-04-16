import joblib
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
import numpy as np

def apply_mitigation():
    model = joblib.load('models/baseline_compliant.pkl')
    test = pd.read_csv('data/silver/test_set.csv')
    
    # Define groups
    age_groups = test['Attribute13'].apply(lambda x: 'Young' if x < 25 else 'Senior')
    
    # Standardize Features to float64 for Fairlearn stability
    protected_cols = [col for col in test.columns if 'Attribute13' in col or 'Attribute9' in col]
    X_test = test.drop(columns=['target'] + protected_cols).values.astype(np.float64)
    
    # CRITICAL: 1 = Good/Approve, 0 = Bad/Deny
    # Statlog labels are 1 (Good) and 2 (Bad). We map them here:
    y_test = test['target'].apply(lambda x: 1 if x == 1 else 0).values

    postprocess_est = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        predict_method='predict_proba'
    )

    # Fit using 1 as the positive outcome
    postprocess_est.fit(X_test, y_test, sensitive_features=age_groups)

    joblib.dump(postprocess_est, 'models/fair_model_optimized.pkl')
    print("✅ Mitigation Re-calibrated: 1=Approve, 0=Deny")

if __name__ == "__main__":
    apply_mitigation()