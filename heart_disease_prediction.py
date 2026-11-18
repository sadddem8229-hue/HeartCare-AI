# =====================================================
# HEART DISEASE PREDICTION
# Student: Sadeem Abdulmohsen
# Instructor: Ms. Latifa Alabdulmohsen
# Academy: SDAIA Academy
# =====================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

PROJECT_NAME = "Heart Disease Prediction"

DATA_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 50)
print(PROJECT_NAME)
print("Student: Sadeem Abdulmohsen")
print("Instructor: Ms. Latifa Alabdulmohsen")
print("Academy: SDAIA Academy")
print("=" * 50)

# 1. Load Dataset
df = pd.read_csv(DATA_URL)

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# 2. Data Cleaning
print("\nMissing Values:", df.isna().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

# 3. Class Balance
counts = df["target"].value_counts().sort_index()

print("\nNo Disease:", counts[0])
print("Disease:", counts[1])

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Class Balance")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("outputs/class_balance.png", dpi=300)
plt.show()
plt.close()

# 4. Feature Distributions
features = ["age", "thalach", "trestbps", "chol", "oldpeak"]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, feature in zip(axes, features):
    sns.histplot(
        data=df,
        x=feature,
        hue="target",
        kde=True,
        ax=ax
    )
    ax.set_title(feature)

plt.tight_layout()
plt.savefig("outputs/feature_distributions.png", dpi=300)
plt.show()
plt.close()

# 5. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png", dpi=300)
plt.show()
plt.close()

# 6. Prepare Data
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 7. Build Models
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    )
}

# 8. Train and Evaluate Models
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results_list = []
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predicted = model.predict(X_test)

    predictions[name] = (probabilities, predicted)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    results_list.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "F1 Score": f1_score(y_test, predicted),
        "ROC AUC": roc_auc_score(y_test, probabilities),
        "CV ROC AUC": cv_scores.mean()
    })

    print("\nModel Trained:", name)

# 9. Model Comparison
results = pd.DataFrame(results_list).round(3)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv("outputs/model_comparison.csv", index=False)

# 10. ROC Curves
plt.figure(figsize=(7, 5))

for name, (probabilities, predicted) in predictions.items():

    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.2f})"
    )

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=300)
plt.show()
plt.close()

# 11. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (name, (probabilities, predicted)) in zip(
    axes,
    predictions.items()
):
    sns.heatmap(
        confusion_matrix(y_test, predicted),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=300)
plt.show()
plt.close()

# 12. Feature Importance
rf_model = models["Random Forest"].named_steps["model"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(
    x=importance.values,
    y=importance.index
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300)
plt.show()
plt.close()

# 13. Save Feature Importance
importance.to_csv("outputs/feature_importance.csv")

print("\nMost Important Feature:", importance.index[0])

# 14. Final Output
print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("All charts and reports saved in outputs folder.")
print("Student: Sadeem Abdulmohsen")
print("=" * 50)

print("\nNote: This project is for educational purposes only.")
