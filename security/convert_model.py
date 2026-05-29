import os
import joblib
import xgboost as xgb

model_path = 'models/fair_model_optimized.pkl'
output_json_path = 'security/credit_model.json'

# Ensure output directory exists
os.makedirs("security", exist_ok=True)

# Try Method 1: Native XGBoost Loader (Most likely cause of the \x0c byte)
try:
    print("Attempting to load via native XGBoost engine...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Save cleanly to JSON
    model.save_model(output_json_path)
    print(f"--> Success! Converted native model to {output_json_path}")

except Exception as native_error:
    print(f"Native XGBoost load skipped: {native_error}")
    
    # Try Method 2: Joblib Loader (If it was dumped via scikit-learn/joblib pipelines)
    try:
        print("\nAttempting to load via Joblib...")
        loaded_obj = joblib.load(model_path)
        print(f"Successfully loaded object type: {type(loaded_obj)}")
        
        # Check if it's the Fairlearn wrapper or raw estimator
        if hasattr(loaded_obj, 'estimator'):
            print("Detected wrapped model structure. Extracting raw XGBoost estimator...")
            base_model = loaded_obj.estimator
        else:
            base_model = loaded_obj
            
        # Extract the underlying native booster if it's an XGBClassifier wrapper
        if hasattr(base_model, 'get_booster'):
            booster = base_model.get_booster()
            booster.save_model(output_json_path)
        else:
            base_model.save_model(output_json_path)
            
        print(f"--> Success! Isolated and saved base model to {output_json_path}")
        
    except Exception as joblib_error:
        print(f"Joblib load failed: {joblib_error}")
        print("\n[Error] Unable to unpack model format. Verify the file isn't corrupted by OneDrive sync states.")
        