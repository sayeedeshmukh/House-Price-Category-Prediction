# House Price Prediction (California Housing Dataset)

Predict house price category (Low / Medium / High) from the California Housing dataset.

## Project Structure

House_Price_Prediction/
├── data/
├── models/
│ ├── best_model.pkl (created after training)
│ └── label_encoder.pkl (created after training)
├── src/
│ ├── preprocessing.py
│ ├── train_models.py
│ └── evaluation.py
├── gui/
│ └── app.py
├── screenshots/ (plots saved here)
├── report.md (export to PDF for submission)
└── requirements.txt

## Setup

1. Create and activate virtual environment

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies

```
pip install -r requirements.txt
```

## Run Steps

1. Preprocess data and generate EDA plots

```
python src/preprocessing.py
```

Outputs:

- `data/processed_california_housing.csv`
- EDA plots saved in `screenshots/`

2. Train, tune, evaluate models and save best

```
python src/train_models.py
```

Outputs:

- `models/best_model.pkl`
- `models/label_encoder.pkl`
- `screenshots/performance_comparison.png`
- `screenshots/confusion_matrix_best.png`
- `screenshots/feature_importance_best.png` (if supported)
- `data/model_metrics.csv`

3. Launch GUI (Streamlit)

```
streamlit run gui/app.py
```

## Notes

- The best model is selected by validation performance (F1-weighted).
- Feature scaling is done using `StandardScaler` inside pipelines for models that need it.
- XGBoost is included; if not installed or no GPU, it will still run on CPU.
- ANN (MLPClassifier) is included as an optional baseline.

## Export Report

Open `report.md` and export to PDF (e.g., from VS Code or any Markdown to PDF tool) as `report.pdf` for submission.




