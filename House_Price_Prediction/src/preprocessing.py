import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df.rename(columns={"MedHouseVal": "MedHouseValue"}, inplace=True)
    return df


def categorize_target(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    # Bin by quantiles to create roughly balanced classes
    labels = ["Low", "Medium", "High"][:n_bins]
    df = df.copy()
    df["PriceCategory"] = pd.qcut(df["MedHouseValue"], q=n_bins, labels=labels)
    return df


def save_distributions(df: pd.DataFrame) -> None:
    for col in df.columns:
        if col == "PriceCategory":
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=40, color="#4C78A8")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(SCREENSHOTS_DIR / f"dist_{col}.png", dpi=150)
        plt.close()


def save_scatter_plots(df: pd.DataFrame, target_col: str = "MedHouseValue") -> None:
    # Limit number of scatter plots for brevity
    feature_subset = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "Population",
        "AveOccup",
    ]
    for col in feature_subset:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df[target_col], s=10, alpha=0.4, color="#F58518")
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.savefig(SCREENSHOTS_DIR / f"scatter_{col}_vs_{target_col}.png", dpi=150)
        plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Checking missing values...")
    missing = df.isna().sum().sum()
    print(f"Total missing values: {missing}")

    print("Saving EDA plots...")
    save_distributions(df)
    save_scatter_plots(df, target_col="MedHouseValue")
    save_correlation_heatmap(df)

    print("Creating target categories (Low/Medium/High)...")
    df_c = categorize_target(df, n_bins=3)

    print("Saving processed dataset...")
    out_csv = DATA_DIR / "processed_california_housing.csv"
    df_c.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()




