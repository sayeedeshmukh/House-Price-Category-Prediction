# House Price Prediction using California Housing Prices Dataset

## 1. Problem Analysis

- **Objective**: Predict house price category (Low / Medium / High) based on features.
- **Dataset**: California Housing Dataset (from scikit-learn). It includes demographic and geographic features aggregated at block group level with target `MedHouseValue` (median house value).
- **ML Problem Type**: Multi-class classification (Low, Medium, High).
- **Significance**: Helps stakeholders understand neighborhood-level affordability and supports decision-making for buyers, sellers, and policy planners.

## 2. Data Preprocessing & EDA

- Loaded dataset via `sklearn.datasets.fetch_california_housing(as_frame=True)`.
- No missing values were found; verified programmatically.
- Converted the continuous target `MedHouseValue` into 3 categories using quantile bins:
  - Low: 0–33rd percentile
  - Medium: 34–66th percentile
  - High: 67–100th percentile
- Visualizations:
  - Distribution plots for each feature
  - Scatter plots vs target (continuous) for selected features
  - Correlation heatmap of features and target
- Feature scaling:
  - Used `StandardScaler` inside ML pipelines for models sensitive to feature scale (LR, KNN, SVM, NB, ANN).

## 3. Models & Training

- Algorithms evaluated:
  - Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Naive Bayes, ANN (MLPClassifier)
- Train/test split: 80/20 with stratification.
- Hyperparameter tuning (GridSearchCV):
  - SVM (C, gamma, kernel)
  - Random Forest (n_estimators, max_depth, min_samples_split)
- Metrics:
  - Accuracy
  - Precision, Recall, F1-Score (weighted)
  - Confusion Matrix
- Comparison:
  - Bar chart comparing model accuracies
- Best model:
  - Selected by validation (cross-validated) F1-weighted score; re-fit on training data and evaluated on hold-out test set.
- Artifacts saved:
  - `models/best_model.pkl`
  - `models/label_encoder.pkl`
  - `data/model_metrics.csv`
  - Figures in `screenshots/`

## 4. Innovation

- Feature Importance:
  - If the best model supports feature importances (e.g., RF, XGB, GB), plotted importance chart.
  - Otherwise used permutation importance as a model-agnostic alternative.
- Explanation of selected model:
  - The best model is documented with rationale, observed performance, and interpretability/operational trade-offs.

## 5. GUI Application (Streamlit)

- Inputs for all features with sensible ranges.
- Predict button triggers model inference and displays the price category with color accents.
- Uses saved `best_model.pkl` and `label_encoder.pkl`.

## 6. How to Reproduce

1. Install requirements (`pip install -r requirements.txt`).
2. Run preprocessing (`python src/preprocessing.py`) to generate processed CSV and plots.
3. Train models (`python src/train_models.py`) to evaluate and save the best model.
4. Launch app (`streamlit run gui/app.py`).

## 7. Results (to be filled after running)

- Test Accuracy (best): \_\_\_\_
- Precision / Recall / F1-weighted: \_**\_ / \_\_** / \_\_\_\_
- Confusion Matrix: see `screenshots/confusion_matrix_best.png`
- Performance chart: see `screenshots/performance_comparison.png`
- Feature importance: see `screenshots/feature_importance_best.png`

## 8. Limitations & Future Work

- Class boundaries depend on quantiles; alternative binning strategies may be explored.
- Geographic effects may benefit from spatial features or engineered interactions.
- Consider calibration of probabilities and more advanced neural networks.

## 9. References

- scikit-learn: California Housing dataset
- XGBoost documentation

