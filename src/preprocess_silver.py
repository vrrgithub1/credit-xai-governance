import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_silver_layer():
    # 1. Load Bronze Data
    df = pd.read_csv('data/bronze/raw_credit_data.csv')
    
    # 2. Map Target (XGBoost likes 0 and 1)
    # Original: 1 = Good, 2 = Bad. Target: 0 = Good, 1 = Default
    df['target'] = df['class'].map({1: 0, 2: 1})
    df = df.drop(columns=['class'])

    # 3. Categorical Encoding (Example of Governance-First Mapping)
    # Mapping specific codes to Ordinal values helps the model understand 'growth'
    employment_map = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}
    df['Attribute7_score'] = df['Attribute7'].map(employment_map)

    # 4. Handle remaining categorical variables
    # We use One-Hot Encoding for non-ordinal features
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_silver = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 5. Save Processed Data
    output_path = 'data/silver'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df_silver.to_csv(f"{output_path}/silver_credit_data.csv", index=False)
    print(f"Silver Layer Created. Shape: {df_silver.shape}")

    # 6. Split for Modeling
    train, test = train_test_split(df_silver, test_size=0.2, random_state=42, stratify=df_silver['target'])
    train.to_csv(f"{output_path}/train_set.csv", index=False)
    test.to_csv(f"{output_path}/test_set.csv", index=False)
    print("Train/Test sets exported to Silver Layer.")

if __name__ == "__main__":
    create_silver_layer()
    