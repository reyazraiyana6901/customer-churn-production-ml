
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import precision_recall_fscore_support

# 1. Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# 2. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 3. Separate features and target
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# 5. Identify feature types
numeric_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]

categorical_features = [
    col for col in X_train.columns
    if col not in numeric_features
]

# 6. Create preprocessing pipeline
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

#7 Combine preprocessing and Logistic Regression


# 7A. Baseline Logistic Regression
baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train baseline model
baseline_model.fit(X_train, y_train)

# Validation predictions
y_val_pred = baseline_model.predict(X_val)
y_val_prob = baseline_model.predict_proba(X_val)[:, 1]

print("\nBASELINE LOGISTIC REGRESSION RESULTS")

print("Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))
print("Precision:", round(precision_score(y_val, y_val_pred), 4))
print("Recall:", round(recall_score(y_val, y_val_pred), 4))
print("F1 Score:", round(f1_score(y_val, y_val_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, y_val_prob), 4))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


# 7B. Class-Weighted Logistic Regression
weighted_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train class-weighted model
weighted_model.fit(X_train, y_train)

# Validation predictions
y_val_pred_weighted = weighted_model.predict(X_val)
y_val_prob_weighted = weighted_model.predict_proba(X_val)[:, 1]

print("\nCLASS-WEIGHTED LOGISTIC REGRESSION RESULTS")

print("Accuracy:", round(accuracy_score(y_val, y_val_pred_weighted), 4))
print("Precision:", round(precision_score(y_val, y_val_pred_weighted), 4))
print("Recall:", round(recall_score(y_val, y_val_pred_weighted), 4))
print("F1 Score:", round(f1_score(y_val, y_val_pred_weighted), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, y_val_prob_weighted), 4))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_weighted))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_weighted))


# 8. Random Forest Classifier
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 9. Train Random Forest
random_forest_model.fit(X_train, y_train)

# 10. Random Forest predictions
y_val_pred_rf = random_forest_model.predict(X_val)
y_val_prob_rf = random_forest_model.predict_proba(X_val)[:, 1]

# 11. Evaluate Random Forest
print("\nRANDOM FOREST RESULTS")

print("Accuracy:", round(accuracy_score(y_val, y_val_pred_rf), 4))
print("Precision:", round(precision_score(y_val, y_val_pred_rf), 4))
print("Recall:", round(recall_score(y_val, y_val_pred_rf), 4))
print("F1 Score:", round(f1_score(y_val, y_val_pred_rf), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, y_val_prob_rf), 4))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_rf))


# 12. Threshold Analysis for Random Forest

print("\nRANDOM FOREST - THRESHOLD ANALYSIS")

thresholds = [0.3, 0.4, 0.5, 0.6]

for threshold in thresholds:

    # Convert probabilities into class predictions
    y_val_pred_threshold = (
        y_val_prob_rf >= threshold
    ).astype(int)

    # Calculate evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        y_val_pred_threshold,
        average="binary",
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))


# 13. Model Comparison

comparison = pd.DataFrame({
    "Model": [
        "Baseline Logistic Regression",
        "Class-Weighted Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
       accuracy_score(y_val, y_val_pred if False else baseline_model.predict(X_val)),
        accuracy_score(y_val, y_val_pred_weighted),
        accuracy_score(y_val, y_val_pred_rf)
    ],
    "Precision": [
        precision_score(y_val, baseline_model.predict(X_val)),
        precision_score(y_val, y_val_pred_weighted),
        precision_score(y_val, y_val_pred_rf)
    ],
    "Recall": [
        recall_score(y_val, baseline_model.predict(X_val)),
        recall_score(y_val, y_val_pred_weighted),
        recall_score(y_val, y_val_pred_rf)
    ],
    "F1 Score": [
        f1_score(y_val, baseline_model.predict(X_val)),
        f1_score(y_val, y_val_pred_weighted),
        f1_score(y_val, y_val_pred_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_val, baseline_model.predict_proba(X_val)[:, 1]),
        roc_auc_score(y_val, y_val_prob_weighted),
        roc_auc_score(y_val, y_val_prob_rf)
    ]
})

print("\nFINAL MODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

# Save comparison results
comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# 14. Final Model Training and Test Evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.base import clone

# Combine training and validation datasets
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

# Create a fresh copy of the Random Forest pipeline
final_model = clone(random_forest_model)

# Train using training + validation data
final_model.fit(X_train_final, y_train_final)

# Predict on untouched test data
y_test_pred = final_model.predict(X_test)
y_test_prob = final_model.predict_proba(X_test)[:, 1]

# Evaluate final model
print("\nFINAL RANDOM FOREST TEST RESULTS")

print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))
print("Precision:", round(precision_score(y_test, y_test_pred), 4))
print("Recall:", round(recall_score(y_test, y_test_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_test_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_prob), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Save the final trained model

model_path = Path("artifacts/random_forest_model.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(final_model, model_path)

print("\nFinal model saved successfully!")
print("Model location:", model_path)