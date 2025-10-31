from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix


def compute_basic_metrics(y_true, y_pred) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }
    return metrics


def plot_confusion(y_true, y_pred, labels: List[str], title: str, out_path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_performance_bar(model_to_metric: Dict[str, float], metric_name: str, out_path) -> None:
    names = list(model_to_metric.keys())
    values = [model_to_metric[k] for k in names]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=names, y=values, palette="Set2")
    plt.ylabel(metric_name)
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Model {metric_name} Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, out_path) -> None:
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_names, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance (Best Model)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()




