
import pandas as pd

# Load the customer churn dataset
df = pd.read_csv(
    "data/raw/Telco-Customer-Churn.csv"
)

# 1. Dataset dimensions
print("Dataset Shape:", df.shape)

# 2. First five records
print("\nFirst 5 Records:")
print(df.head())

# 3. Column information
print("\nDataset Information:")
df.info()

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# 6. Duplicate records
print("\nDuplicate Rows:", df.duplicated().sum())

# 7. Target variable distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# 8. Target variable percentage
print("\nChurn Percentage:")
print(
    df["Churn"].value_counts(normalize=True) * 100
)

import pandas as pd

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

missing_records = df[df["TotalCharges"].isnull()]

print("Missing TotalCharges Records:")
print(
    missing_records[
        ["customerID", "tenure", "MonthlyCharges", "Churn"]
    ]
)