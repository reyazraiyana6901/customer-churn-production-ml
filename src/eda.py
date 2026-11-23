
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Create output directory
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------
# 1. Churn Distribution
# --------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(output_dir / "churn_distribution.png")
plt.show()


# --------------------------------
# 2. Customer Tenure Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    element="step"
)

plt.title("Customer Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(output_dir / "tenure_distribution.png")
plt.show()


# --------------------------------
# 3. Contract Type vs Churn
# --------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)

plt.title("Churn Distribution Across Contract Types")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")

plt.tight_layout()
plt.savefig(output_dir / "contract_vs_churn.png")
plt.show()


# --------------------------------
# 4. Monthly Charges vs Churn
# --------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig(output_dir / "monthly_charges_vs_churn.png")
plt.show()


# --------------------------------
# 5. Investigate Missing Records
# --------------------------------

missing_records = df[df["TotalCharges"].isnull()]

print("\nMissing TotalCharges Records:")
print(
    missing_records[
        ["customerID", "tenure", "MonthlyCharges", "Churn"]
    ]
)

print("\nEDA completed successfully!")

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    element="step"
)

plt.title("Customer Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(output_dir / "tenure_distribution.png")
plt.show()


# 5. Categorical Features vs Churn

categorical_features = [
    "InternetService",
    "PaymentMethod",
    "TechSupport"
]

for feature in categorical_features:

    churn_percentage = pd.crosstab(
        df[feature],
        df["Churn"],
        normalize="index"
    ) * 100

    churn_percentage.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5)
    )

    plt.title(f"Churn Distribution Across {feature}")
    plt.xlabel(feature)
    plt.ylabel("Percentage of Customers")
    plt.xticks(rotation=20)
    plt.legend(title="Churn")
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{feature.lower()}_vs_churn.png"
    )

    plt.show()

    print(f"\nChurn Percentage by {feature}:")
    print(churn_percentage.round(2))
