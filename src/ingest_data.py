import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def ingest_bronze_data():
    # 1. Fetch Dataset (Statlog German Credit Data ID is 144)
    print("Fetching data from UCI Repository...")
    statlog_german_credit_data = fetch_ucirepo(id=144)
    
    # 2. Extract features and targets
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets
    
    # 3. Combine into a single DataFrame for the Bronze layer
    bronze_df = pd.concat([X, y], axis=1)
    
    # 4. Create directory if it doesn't exist
    output_path = 'data/bronze'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 5. Save to CSV
    file_name = f"{output_path}/raw_credit_data.csv"
    bronze_df.to_csv(file_name, index=False)
    print(f"Successfully saved Bronze data to {file_name}")
    print(f"Dataset Shape: {bronze_df.shape}")

if __name__ == "__main__":
    ingest_bronze_data()
