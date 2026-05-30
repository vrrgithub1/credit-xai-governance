import joblib
import os

# 1. Load the Fairlearn wrapped model
model_path = 'models/fair_model_optimized.pkl'

try:
    model = joblib.load(model_path)
    print(f"Successfully loaded: {type(model)}")
    
    # 2. Check if we can access the underlying XGBoost model
    # ThresholdOptimizer usually stores the base model in .estimator
    if hasattr(model, 'estimator'):
        print("Base XGBoost model detected inside the wrapper.")
    
    # 3. Instead of converting to JSON, we will use the live object for ART
    # We'll save a clean 'security' version using joblib dump
    os.makedirs("security", exist_ok=True)
    joblib.dump(model, "security/fair_credit_model.pkl")
    print("Saved optimized fair model to security/fair_credit_model.pkl")

except Exception as e:
    print(f"Loading failed: {e}")