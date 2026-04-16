import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_mitigation():
    # 1. Load Data and Model
    test = pd.read_csv('data/silver/test_set.csv')
    model_fair = joblib.load('models/fair_model_optimized.pkl')
    
    # 2. Re-prepare the feature matrix (consistent with validation)
    protected_cols = [col for col in test.columns if 'Attribute13' in col or 'Attribute9' in col]
    X_test_df = test.drop(columns=['target'] + protected_cols)
    
    # 3. Initialize SHAP Explainer
    # We explain the underlying XGBoost model ('estimator')
    explainer = shap.TreeExplainer(model_fair.estimator)
    shap_values = explainer.shap_values(X_test_df)

    # 4. Generate Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.title("SHAP Feature Importance: Optimized Fair Model")
    
    # Save the results for the governance report
    plt.savefig('reports/fair_model_shap_summary.png', bbox_inches='tight')
    plt.show()
    
    print("✅ SHAP Analysis Complete. Plot saved to: reports/fair_model_shap_summary.png")

if __name__ == "__main__":
    explain_mitigation()