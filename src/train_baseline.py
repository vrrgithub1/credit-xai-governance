import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

def train_baselines():
    # 1. Load Silver Data
    train = pd.read_csv('data/silver/train_set.csv')
    test = pd.read_csv('data/silver/test_set.csv')

    # Define Target
    y_train = train['target']
    y_test = test['target']
    
    # Identify Protected Attributes (Update names based on your Silver columns)
    # Usually, Attribute 13 is Age, and Attribute 9 is Gender/Status
    protected_cols = [col for col in train.columns if 'Attribute13' in col or 'Attribute9' in col]

    # --- MODEL A: The Full Baseline (Unfiltered) ---
    X_train_a = train.drop(columns=['target'])
    X_test_a = test.drop(columns=['target'])
    
    model_a = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model_a.fit(X_train_a, y_train)
    
    # --- MODEL B: The Compliant Baseline (Dropping Protected Attributes) ---
    X_train_b = X_train_a.drop(columns=protected_cols)
    X_test_b = X_test_a.drop(columns=protected_cols)
    
    model_b = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model_b.fit(X_train_b, y_train)

    # 2. Evaluate & Compare
    for name, model, X_test in [("Full Baseline", model_a, X_test_a), ("Compliant Baseline", model_b, X_test_b)]:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"\n--- {name} Results ---")
        print(f"ROC-AUC Score: {auc:.4f}")
        
    # 3. Save Models to Gold Layer
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model_a, 'models/baseline_full.pkl')
    joblib.dump(model_b, 'models/baseline_compliant.pkl')
    print("\nModels saved to /models folder.")

if __name__ == "__main__":
    train_baselines()
    