"""
House Price Prediction using California Housing Prices Dataset (Classification)

This script performs a complete ML mini-project:
 - Loads the California Housing dataset
 - EDA with visualizations saved as PNGs
 - Converts the continuous target into 3 categories: Low / Medium / High
 - Preprocesses data and evaluates multiple classifiers
 - Runs hyperparameter tuning for SVM and RandomForest (and XGBoost if available)
 - Compares models with metrics and plots
 - Saves the best model pipeline to best_model.pkl for use by gui_app.py

Run directly:
  python house_price_model.py

Outputs written to the current directory:
  - best_model.pkl (model artifact for GUI)
  - figures/*.png (EDA and evaluation plots)
  - results_metrics.csv (table of scores per model)
"""

import os
import warnings
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


# Optional XGBoost
try:
    from xgboost import XGBClassifier  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


warnings.filterwarnings("ignore")


def ensure_output_dirs() -> str:
    figures_dir = os.path.join(os.getcwd(), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    dataset = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = dataset.data.copy()
    y_cont: pd.Series = dataset.target.copy()  # continuous target (MedHouseVal)
    feature_names: List[str] = list(dataset.feature_names)
    return X, y_cont, feature_names


def bin_target_into_categories(y_cont: pd.Series) -> pd.Series:
    # Create 3 balanced categories using tertiles
    bins = pd.qcut(y_cont, q=3, labels=["Low", "Medium", "High"])  # type: ignore
    return bins.astype(str)


def run_eda(X: pd.DataFrame, y_cont: pd.Series, y_cat: pd.Series, figures_dir: str) -> None:
    """Generate and save core EDA plots."""
    # Basic describe tables
    desc_num = X.describe().T
    desc_num.to_csv(os.path.join(figures_dir, "features_describe.csv"))

    # Target (continuous) distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(y_cont, kde=True, bins=40, color="#4C72B0")
    plt.title("Distribution of Continuous Target (MedHouseVal)")
    plt.xlabel("MedHouseVal (in $100k)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "target_continuous_distribution.png"), dpi=150)
    plt.close()

    # Target (categorical) class counts
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_cat, order=["Low", "Medium", "High"], palette="viridis")
    plt.title("Target Class Counts (Low/Medium/High)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "target_class_counts.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    corr = X.corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", annot=False, square=True, cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Feature histograms grid
    cols = X.columns
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(cols):
        sns.histplot(X[col], kde=False, bins=40, ax=axes[idx], color="#55A868")
        axes[idx].set_title(f"Histogram: {col}")
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "feature_histograms.png"), dpi=150)
    plt.close()


def build_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="auto"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )
    return models


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, figures_dir: str) -> Dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    # Confusion matrix plot
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(figures_dir, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close()

    return {
        "model": name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "pipeline": pipeline,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def hyperparameter_tuning(models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return param grids keyed by model name for GridSearchCV on the pipeline's 'clf__' params."""
    grids: Dict[str, Dict[str, Any]] = {
        "SVM": {
            "clf__C": [0.5, 1.0, 2.0],
            "clf__gamma": ["scale", 0.05, 0.1],
            "clf__kernel": ["rbf"],
        },
        "Random Forest": {
            "clf__n_estimators": [150, 300, 500],
            "clf__max_depth": [None, 8, 12],
            "clf__max_features": ["auto", "sqrt", 0.7],
        },
    }
    if XGBOOST_AVAILABLE and ("XGBoost" in models):
        grids["XGBoost"] = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
        }
    return grids


def main() -> None:
    figures_dir = ensure_output_dirs()

    print("Loading dataset...")
    X, y_cont, feature_names = load_dataset()
    y_cat = bin_target_into_categories(y_cont)

    print("Running EDA and saving figures...")
    run_eda(X, y_cont, y_cat, figures_dir)

    print("Preparing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    print("Building models...")
    base_models = build_models()

    # Fit base models
    results: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, est in base_models.items():
        print(f"Training base model: {name}")
        pipeline = build_pipeline(est)
        pipeline.fit(X_train, y_train)
        res = evaluate_model(name, pipeline, X_test, y_test, figures_dir)
        results.append(res)
        fitted_pipelines[name] = pipeline

    # Hyperparameter tuning (SVM, RandomForest, XGBoost if available)
    print("Starting hyperparameter tuning (this may take a few minutes)...")
    grids = hyperparameter_tuning(base_models)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    tuned_results: List[Dict[str, Any]] = []
    for name, grid in grids.items():
        print(f"Grid search for: {name}")
        pipeline = build_pipeline(base_models[name])
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_pipeline: Pipeline = gs.best_estimator_
        print(f"  Best params for {name}: {gs.best_params_}")
        res = evaluate_model(f"{name} (Tuned)", best_pipeline, X_test, y_test, figures_dir)
        res["best_params"] = gs.best_params_
        tuned_results.append(res)
        fitted_pipelines[f"{name} (Tuned)"] = best_pipeline

    all_results = results + tuned_results

    # Results table
    results_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Precision (macro)": r["precision_macro"],
            "Recall (macro)": r["recall_macro"],
            "F1 (macro)": r["f1_macro"],
        }
        for r in all_results
    ])
    results_df.sort_values(by=["Accuracy", "F1 (macro)"], ascending=[False, False], inplace=True)
    results_df.to_csv("results_metrics.csv", index=False)

    # Model comparison bar chart (Accuracy)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="Accuracy", palette="mako")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Model Comparison (Accuracy)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison_accuracy.png"), dpi=150)
    plt.close()

    # Determine best model by Accuracy then F1
    best_row = results_df.iloc[0]
    best_model_name = str(best_row["Model"])
    best_pipeline = fitted_pipelines[best_model_name]

    # Persist artifact with metadata used by GUI
    artifact = {
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "class_labels": ["Low", "Medium", "High"],
        "best_model_name": best_model_name,
        "results": results_df.to_dict(orient="records"),
    }
    joblib.dump(artifact, "best_model.pkl")

    print("\n===== SUMMARY =====")
    print(results_df.to_string(index=False))
    print("\nBest model:", best_model_name)
    # Short conclusion
    conclusion = (
        f"Based on held-out test performance, the selected model is '{best_model_name}', "
        "which achieved the highest accuracy among evaluated candidates. "
        "Tree-ensemble methods and tuned SVM typically perform well on tabular data; "
        "here, the chosen model provided the best generalization for classifying prices into Low/Medium/High."
    )
    print("\nConclusion:")
    print(conclusion)


if __name__ == "__main__":
    main()


