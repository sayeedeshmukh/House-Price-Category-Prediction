import warnings
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import (
    compute_basic_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_performance_bar,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COLUMN = "PriceCategory"


def build_models() -> Dict[str, Pipeline]:
    scale_then = [
        ("scaler", StandardScaler()),
    ]

    models: Dict[str, Pipeline] = {
        "LogReg": Pipeline(scale_then + [("clf", LogisticRegression(max_iter=1000, n_jobs=None))]),
        "KNN": Pipeline(scale_then + [("clf", KNeighborsClassifier())]),
        "SVM": Pipeline(scale_then + [("clf", SVC(probability=True))]),
        "GaussianNB": Pipeline(scale_then + [("clf", GaussianNB())]),
        "DecisionTree": Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
        "RandomForest": Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        "GradientBoosting": Pipeline([("clf", GradientBoostingClassifier(random_state=42))]),
    }

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        tree_method="hist",
                    ),
                )
            ]
        )
    except Exception:
        pass

    return models


def build_param_grids() -> Dict[str, Dict]:
    grids: Dict[str, Dict] = {
        # Keep grids compact to control runtime
        "SVM": {
            "clf__C": [0.1, 1.0],
            "clf__gamma": ["scale", 0.01],
            "clf__kernel": ["rbf"],
        },
        "RandomForest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 15],
            "clf__min_samples_split": [2, 5],
        },
        "KNN": {
            "clf__n_neighbors": [5, 7, 9],
            "clf__weights": ["uniform", "distance"],
        },
        "LogReg": {
            "clf__C": [0.1, 1.0],
            "clf__solver": ["lbfgs"],
            "clf__penalty": ["l2"],
        },
        "GradientBoosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
        "DecisionTree": {
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
    }
    return grids


def main():
    # Load processed data
    data_path = DATA_DIR / "processed_california_housing.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. Run preprocessing.py first."
        )

    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str).copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    models = build_models()
    grids = build_param_grids()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_metrics: Dict[str, Dict[str, float]] = {}
    model_objects: Dict[str, Pipeline] = {}

    # Train baseline models only (no tuning here)
    for name, pipeline in models.items():
        print(f"Training {name}...")
        model = pipeline.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = compute_basic_metrics(y_test, y_pred)
        model_metrics[name] = metrics
        model_objects[name] = model
        print(f"{name} metrics: {metrics}")

    # Determine best baseline model by F1-weighted
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.sort_values("f1_weighted", ascending=False, inplace=True)
    best_name = metrics_df.index[0]
    best_model = model_objects[best_name]
    print(f"Best baseline model: {best_name}")

    # Hyperparameter tuning ONLY for the best baseline model
    if best_name in grids:
        print(f"Tuning best model: {best_name}")
        grid = grids[best_name]
        gs = GridSearchCV(
            estimator=models[best_name],
            param_grid=grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        print(f"{best_name} tuned best params: {gs.best_params_}")
        # Evaluate tuned best on test set and replace metrics
        y_best_pred = best_model.predict(X_test)
        tuned_metrics = compute_basic_metrics(y_test, y_best_pred)
        model_metrics[best_name] = tuned_metrics
        model_objects[best_name] = best_model

        # Recompute metrics df after tuning update
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df.sort_values("f1_weighted", ascending=False, inplace=True)

    # Save metrics table after potential tuning
    metrics_csv = DATA_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_csv)
    print(f"Saved metrics: {metrics_csv}")

    # Save performance comparison chart
    perf_chart = SCREENSHOTS_DIR / "performance_comparison.png"
    plot_performance_bar(
        {k: v["accuracy"] for k, v in model_metrics.items()},
        metric_name="Accuracy",
        out_path=perf_chart,
    )

    # Save confusion matrix for best model
    class_names = list(label_encoder.classes_)
    y_best_pred = best_model.predict(X_test)
    cm_path = SCREENSHOTS_DIR / "confusion_matrix_best.png"
    plot_confusion(y_test, y_best_pred, labels=list(range(len(class_names))), title=f"Confusion Matrix - {best_name}", out_path=cm_path)

    # Feature importance if available
    try:
        importances = None
        if hasattr(best_model, "named_steps") and "clf" in best_model.named_steps:
            clf = best_model.named_steps["clf"]
        else:
            clf = best_model

        if hasattr(clf, "feature_importances_"):
            importances = np.asarray(clf.feature_importances_)
        elif hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_)
            importances = np.mean(np.abs(coef), axis=0)

        if importances is not None:
            fi_path = SCREENSHOTS_DIR / "feature_importance_best.png"
            plot_feature_importance(FEATURE_COLUMNS, importances, fi_path)
    except Exception:
        pass

    # Persist best model and label encoder
    best_model_path = MODELS_DIR / "best_model.pkl"
    label_enc_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(best_model, best_model_path)
    joblib.dump(label_encoder, label_enc_path)
    print(f"Saved best model to {best_model_path}")
    print(f"Saved label encoder to {label_enc_path}")


if __name__ == "__main__":
    main()


