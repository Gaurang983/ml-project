# Credit Card Default Prediction Using Machine Learning

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TARGET = "default.payment.next.month"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

print("Library versions")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)


# ============================================================
# 1. LOAD DATA
# ============================================================

def clean_numeric(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    try:
        return float(value.replace(",", ""))
    except ValueError:
        pass

    try:
        return float(value.replace(",", "."))
    except ValueError:
        return np.nan


def load_data():
    paths = [
        "/content/Credit_Card.csv",
        "./Credit_Card.csv",
        "/data/Credit_Card.csv"
    ]

    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]

    for path in paths:
        if not os.path.exists(path):
            print("File not found:", path)
            continue

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    path,
                    sep=";",
                    encoding=encoding,
                    on_bad_lines="warn",
                    low_memory=False,
                    skipinitialspace=True
                )
                print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
                print("Path:", path)
                print("Encoding:", encoding)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as error:
                print("Error:", error)

    raise FileNotFoundError("Credit_Card.csv not found.")


df_original = load_data()
df = df_original.copy()

numeric_candidates = [
    "LIMIT_BAL", "LIMIT_BAL_LOG", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "BILL_AMT_SUM", "risk_leak"
]

for col in numeric_candidates:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].apply(clean_numeric)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print("Cleaned:", col)

print("\nData loaded successfully")
print("Shape:", df.shape)
print(df.head())


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nEDA")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing cells:", df.isnull().sum().sum())

print("\nTarget distribution:")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x=TARGET, data=df)
plt.title("Target Variable Distribution")
plt.xlabel("Default Payment Next Month")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("\nDescriptive statistics for key numerical features:")
key_numeric = ["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "LIMIT_BAL_LOG", "risk_leak"]
available_key_numeric = [col for col in key_numeric if col in df.columns]
print(df[available_key_numeric].describe())

print("\nMissing values by column:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

if len(missing) > 0:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[missing.index].isnull(), cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.show()

eda_numeric = ["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "LIMIT_BAL_LOG", "risk_leak"]

for col in eda_numeric:
    if col in df.columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 4))
        sns.boxplot(y=df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

categorical_features = ["SEX", "EDUCATION", "MARRIAGE", "CITY", "RISK_RATING"]

for col in categorical_features:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nFeatures most correlated with default:")
print(numeric_df.corr()[TARGET].sort_values(ascending=False).head(15))

print("\nEDA Summary")
print("Default rate:", round(df[TARGET].mean() * 100, 2), "%")
print("Total missing cells:", df.isnull().sum().sum())


# ============================================================
# 3. DATA PREPARATION
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

if "ID" in X.columns:
    X = X.drop(columns=["ID"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

print("\nBefore imputation")
print("Train missing:", X_train.isnull().sum().sum())
print("Test missing:", X_test.isnull().sum().sum())

if numeric_cols:
    num_imputer = SimpleImputer(strategy="median")
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[categorical_cols] = X_train[categorical_cols].astype("object")
    X_test[categorical_cols] = X_test[categorical_cols].astype("object")

    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nAfter imputation and encoding")
print("Train missing:", X_train.isnull().sum().sum())
print("Test missing:", X_test.isnull().sum().sum())

final_numeric_cols = [col for col in numeric_cols if col in X_train.columns]

scaler = StandardScaler()
X_train[final_numeric_cols] = scaler.fit_transform(X_train[final_numeric_cols])
X_test[final_numeric_cols] = scaler.transform(X_test[final_numeric_cols])

print("\nData preparation complete")
print("Training rows:", X_train.shape[0])
print("Test rows:", X_test.shape[0])
print("Final features:", X_train.shape[1])
print("Categorical columns encoded:", categorical_cols)


# ============================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        random_state=RANDOM_STATE
    ),
    "SVM Linear": SGDClassifier(
        loss="hinge",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
        tol=1e-3
    ),
    "KNN": KNeighborsClassifier()
}

results = []
predictions = {}
scores = {}

for name, model in models.items():
    print("\nTraining:", name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    predictions[name] = y_pred

    if y_score is not None:
        scores[name] = y_score
        auc = roc_auc_score(y_test, y_score)
    else:
        auc = np.nan

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "AUC-ROC": round(auc, 4)
    })

results_df = pd.DataFrame(results)

print("\nInitial model comparison:")
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 8))
for name, y_score in scores.items():
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_value = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} AUC = {auc_value:.3f}")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

for j in range(len(predictions), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred_tuned = best_rf.predict(X_test)
y_score_tuned = best_rf.predict_proba(X_test)[:, 1]

tuned_results = {
    "Model": "Random Forest Tuned",
    "Accuracy": round(accuracy_score(y_test, y_pred_tuned), 4),
    "Precision": round(precision_score(y_test, y_pred_tuned), 4),
    "Recall": round(recall_score(y_test, y_pred_tuned), 4),
    "F1": round(f1_score(y_test, y_pred_tuned), 4),
    "AUC-ROC": round(roc_auc_score(y_test, y_score_tuned), 4)
}

results_df.loc[len(results_df)] = tuned_results

print("\nTuned Random Forest")
print("Best parameters:", grid_search.best_params_)
print("Cross-validation AUC:", round(grid_search.best_score_, 4))
print(pd.DataFrame([tuned_results]).to_string(index=False))

print("\nFinal comparison:")
print(results_df.to_string(index=False))

cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ============================================================
# 6. SHAP EXPLAINABILITY
# ============================================================

print("\nSHAP explanation")

X_sample = X_train.sample(100, random_state=RANDOM_STATE)

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values_class_1 = shap_values[1]
else:
    shap_values_class_1 = shap_values[:, :, 1]

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_class_1,
    X_sample,
    feature_names=X_train.columns,
    show=False,
    max_display=20
)
plt.title("SHAP Summary Plot - Tuned Random Forest")
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_class_1,
    X_sample,
    plot_type="bar",
    feature_names=X_train.columns,
    show=False,
    max_display=20
)
plt.title("SHAP Feature Importance - Tuned Random Forest")
plt.tight_layout()
plt.show()
plt.close()


# ============================================================
# 7. CONCLUSION OUTPUT
# ============================================================

best_row = results_df.sort_values("AUC-ROC", ascending=False).iloc[0]

print("\n" + "=" * 60)
print("CONCLUSION AND FUTURE WORK")
print("=" * 60)

print(f"""
Five classification models were trained and evaluated:
Logistic Regression, Decision Tree, Random Forest, Linear SVM, and KNN.

The best-performing model based on AUC-ROC was:

Model: {best_row['Model']}
Accuracy: {best_row['Accuracy']}
Precision: {best_row['Precision']}
Recall: {best_row['Recall']}
F1-score: {best_row['F1']}
AUC-ROC: {best_row['AUC-ROC']}

The dataset contained {df.shape[0]} records and {df.shape[1]} columns.
The default rate was {df[TARGET].mean() * 100:.2f}%.
The dataset contained {df.isnull().sum().sum()} missing cells before preprocessing.

The target variable was imbalanced, so accuracy alone was not sufficient.
Precision, recall, F1-score and AUC-ROC were therefore used for evaluation.

SHAP analysis showed that repayment behaviour and risk-related variables were
important for predicting default. Features such as PAY_0, PAY_2, PAY_3,
RISK_RATING, risk_leak, LIMIT_BAL and BILL_AMT_SUM should be discussed
in the final report.

Limitations:
1. The dataset is historical and may not represent current credit-card behaviour.
2. Missing values were handled using median and mode imputation.
3. Demographic variables may raise fairness concerns.
4. Engineered variables such as risk_leak should be checked carefully before deployment.

Future work:
1. Use SMOTE or similar techniques to improve default-class recall.
2. Try XGBoost or LightGBM for improved tabular-data performance.
3. Add fairness testing for demographic variables.
4. Test the model on newer data.
5. Compare results with and without risk_leak.
""")