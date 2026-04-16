import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

def run_shap_audit():
    # 1. Load the Compliant Model and Test Data
    model = joblib.load('models/baseline_compliant.pkl')
    test = pd.read_csv('data/silver/test_set.csv')
    
    # Define features (Must match the training set for Model B)
    # Ensure target and protected attributes are dropped
    protected_cols = [col for col in test.columns if 'Attribute13' in col or 'Attribute9' in col]
    X_test = test.drop(columns=['target'] + protected_cols)

    # 2. Initialize SHAP Explainer
    # TreeExplainer is optimized for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 3. Generate Global Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    
    # Save the audit plot to a new 'reports' directory
    if not os.path.exists('reports'): os.makedirs('reports')
    plt.title("Phase III: Global Feature Importance (Compliant Model)")
    plt.savefig('reports/shap_summary_compliant.png', bbox_inches='tight')
    plt.show()
    
    print("✅ SHAP Audit Complete. Summary plot saved to reports/shap_summary_compliant.png")

if __name__ == "__main__":
    run_shap_audit()
    