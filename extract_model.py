"""
Extract real logistic regression model weights from the trained model
and export them for use in the JavaScript dashboard.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# ── Load data (same path as in the notebook) ──────────────────────────
df = pd.read_csv(r"C:\Users\babba\heart_disease_risk_dataset_earlymed.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── Prepare features & target (exactly as in notebook) ────────────────
x = df.drop(columns=['Heart_Risk'])
y = df['Heart_Risk']

# NOTE: notebook splits with (x, y, test_size=0.35, random_state=42)
# but passes them REVERSED: x_test, x_train, y_test, y_train = train_test_split(...)
# Replicating the EXACT same split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, test_size=0.35, random_state=42
)

print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

# ── Train model ────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────
train_pred = lr.predict(x_train)
test_pred  = lr.predict(x_test)
train_acc  = accuracy_score(y_train, train_pred)
test_acc   = accuracy_score(y_test,  test_pred)
cv_scores  = cross_val_score(lr, x_train, y_train, cv=5)
conf_mat   = confusion_matrix(y_test, test_pred)

print(f"\nTraining Accuracy : {train_acc:.6f}")
print(f"Testing  Accuracy : {test_acc:.6f}")
print(f"CV Score (mean)   : {cv_scores.mean():.6f}")
print(f"\nConfusion Matrix:\n{conf_mat}")

# ── Extract coefficients ───────────────────────────────────────────────
feature_names = list(x.columns)
coefficients  = lr.coef_[0].tolist()
intercept     = float(lr.intercept_[0])

coef_dict = dict(zip(feature_names, coefficients))

print(f"\nIntercept: {intercept:.6f}")
print("\nCoefficients:")
for name, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:30s}: {coef:+.6f}")

# ── Build JSON export ──────────────────────────────────────────────────
model_data = {
    "model_type": "Logistic Regression",
    "intercept": round(intercept, 8),
    "coefficients": {k: round(v, 8) for k, v in coef_dict.items()},
    "feature_order": feature_names,
    "metrics": {
        "train_accuracy": round(float(train_acc), 6),
        "test_accuracy":  round(float(test_acc),  6),
        "cv_score":       round(float(cv_scores.mean()), 6),
        "error_rate":     round(float(1 - test_acc), 6)
    },
    "confusion_matrix": {
        "TN": int(conf_mat[0][0]),
        "FP": int(conf_mat[0][1]),
        "FN": int(conf_mat[1][0]),
        "TP": int(conf_mat[1][1])
    },
    "dataset": {
        "total_records": len(df),
        "n_features": len(feature_names),
        "positive_rate": round(float(y.mean()), 4),
        "age_min": int(df['Age'].min()),
        "age_max": int(df['Age'].max()),
        "age_mean": round(float(df['Age'].mean()), 2)
    }
}

output_path = r"d:\disease prediction\model_weights.json"
with open(output_path, "w") as f:
    json.dump(model_data, f, indent=2)

print(f"\n✅ Model weights exported to: {output_path}")
print(f"   Intercept: {intercept:.6f}")
print(f"   Features: {len(feature_names)}")
print(f"   Test Accuracy: {test_acc*100:.4f}%")
