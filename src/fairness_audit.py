import pandas as pd
import joblib
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, demographic_parity_ratio

def run_fairness_audit():
    # 1. Load Model and Test Data
    model = joblib.load('models/baseline_compliant.pkl')
    test = pd.read_csv('data/silver/test_set.csv')
    
    # 2. Reconstruct the 'Age' group for auditing
    # We use Attribute13 (Age) from the test set. 
    # Let's define "Younger" as < 25 as per your GOVERNANCE.md
    test['age_group'] = test['Attribute13'].apply(lambda x: 'Young' if x < 25 else 'Senior')
    
    # 3. Get Predictions (0 = Good/Approve, 1 = Bad/Deny)
    # Fairlearn usually measures the 'positive' outcome (Approval), so we flip the target
    protected_cols = [col for col in test.columns if 'Attribute13' in col or 'Attribute9' in col or col == 'age_group']
    X_test = test.drop(columns=['target'] + protected_cols)
    
    preds = model.predict(X_test)
    approvals = [1 if p == 0 else 0 for p in preds] # 1 is Approved, 0 is Denied

    # 4. Calculate Fairness Metrics
    metrics = MetricFrame(
        metrics=selection_rate,
        y_true=[1 if t == 0 else 0 for t in test['target']],
        y_pred=approvals,
        sensitive_features=test['age_group']
    )

    print("--- Fairness Audit: Age Groups (<25 vs 25+) ---")
    print(f"Selection Rates:\n{metrics.by_group}")
    
    dp_ratio = demographic_parity_ratio(y_true=[1 if t == 0 else 0 for t in test['target']], 
                                        y_pred=approvals, 
                                        sensitive_features=test['age_group'])
    
    print(f"\nDemographic Parity Ratio: {dp_ratio:.4f}")
    
    if dp_ratio < 0.80:
        print("⚠️ WARNING: Disparate Impact detected (Ratio < 0.80). Model favors the Senior group.")
    else:
        print("✅ PASS: Model meets the 4/5ths rule for fairness.")

if __name__ == "__main__":
    run_fairness_audit()
    