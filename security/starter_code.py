import pandas as pd
import xgboost as xgb
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import HopSkipJump

# 1. Load your existing model and a sample of 'Denied' records
# (Assuming your model is saved as credit_model.json)
model = xgb.XGBClassifier()
model.load_model("credit_model.json")

# 2. Wrap the model in the ART interface
# nb_features should match your Credit XAI feature count (e.g., 47)
art_classifier = XGBoostClassifier(model=model, nb_features=47)

# 3. Define the Attack
# HopSkipJump is great for tabular data. 
# targeted=False means we just want to flip the result (from 0 to 1).
attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=10, max_eval=100)

print("Red Team Environment Initialized. Ready to generate adversarial samples.")
