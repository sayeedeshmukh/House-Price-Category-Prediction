# House Price Prediction using California Housing Dataset

A complete Machine Learning mini-project that predicts house price categories (Low/Medium/High) using the California Housing Prices dataset.

## Features

- **Complete ML Pipeline**: EDA, preprocessing, multiple classification algorithms
- **8 Classification Models**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Naive Bayes
- **Hyperparameter Tuning**: Optimized SVM, Random Forest, and XGBoost
- **Interactive GUI**: Streamlit web app for easy predictions
- **Performance Analysis**: Accuracy, precision, recall, F1-score, confusion matrices

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python house_price_model.py
```

This will:

- Load the California Housing dataset
- Perform EDA and save visualizations
- Train multiple classification models
- Perform hyperparameter tuning
- Save the best model as `best_model.pkl`

### 3. Launch the GUI

```bash
streamlit run gui_app.py
```

## Files

- `house_price_model.py` - Complete ML pipeline and model training
- `gui_app.py` - Streamlit web interface for predictions
- `requirements.txt` - Python dependencies
- `best_model.pkl` - Trained model artifact (generated after training)
- `figures/` - EDA and evaluation plots (generated after training)
- `results_metrics.csv` - Model performance comparison (generated after training)

## Model Performance

The script evaluates all models and selects the best performing one based on accuracy and F1-score. Typical results show tree-based ensemble methods (Random Forest, XGBoost) performing well on this tabular dataset.

## Deployment

For Streamlit Cloud deployment:

1. Push this repository to GitHub
2. Connect to Streamlit Cloud
3. Deploy with the repository URL
4. The app will automatically install dependencies and run

## Dataset

Uses the California Housing dataset from scikit-learn with 8 features:

- MedInc: Median income
- HouseAge: House age
- AveRooms: Average rooms per household
- AveBedrms: Average bedrooms per household
- Population: Block group population
- AveOccup: Average occupants per household
- Latitude: Geographic latitude
- Longitude: Geographic longitude

The continuous target (median house value) is converted into 3 balanced categories using tertiles.
