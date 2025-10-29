# California Housing Price Prediction

A machine learning project that predicts house price categories (Low/Medium/High) using the California Housing dataset. The project includes a complete ML pipeline with EDA, model training, hyperparameter tuning, and a Streamlit web application.

## 🏠 Project Overview

This project converts the continuous target variable (median house value) into three balanced categories and trains multiple classifiers to predict these categories. The best performing model is then deployed as an interactive Streamlit web application.

## 📊 Features

- **Exploratory Data Analysis (EDA)**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, and XGBoost
- **Hyperparameter Tuning**: Grid search optimization for SVM, Random Forest, and XGBoost
- **Model Comparison**: Detailed evaluation metrics and confusion matrices
- **Interactive Web App**: Streamlit-based GUI for real-time predictions
- **Model Persistence**: Best model saved as pickle file for deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/california-housing-prediction.git
cd california-housing-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. **Train the model**:

```bash
python house_price_model.py
```

This will:

- Load the California Housing dataset
- Perform EDA and save visualizations to `figures/` directory
- Train multiple models and perform hyperparameter tuning
- Save the best model as `best_model.pkl`
- Generate evaluation metrics in `results_metrics.csv`

2. **Launch the web application**:

```bash
streamlit run gui_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
├── house_price_model.py      # Main ML training script
├── gui_app.py               # Streamlit web application
├── requirements.txt         # Python dependencies
├── best_model.pkl          # Trained model (generated after training)
├── results_metrics.csv     # Model evaluation results
├── figures/                # EDA and evaluation visualizations
│   ├── confusion_matrix_*.png
│   ├── correlation_heatmap.png
│   ├── feature_histograms.png
│   └── model_comparison_accuracy.png
├── assignment/             # React components (separate project)
└── House_Price_Prediction/ # Alternative implementation
```

## 🎯 Model Performance

The project evaluates multiple models and selects the best performer based on accuracy and F1-score. Typical results show:

- **Random Forest** and **XGBoost** often achieve the highest accuracy
- **SVM** with tuning performs well on this dataset
- All models are evaluated using stratified cross-validation

## 🌐 Streamlit Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `gui_app.py`
5. Deploy!

### Local Development

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run gui_app.py
```

## 📈 Dataset Information

The California Housing dataset contains 8 features:

- **MedInc**: Median income in block group (in $10k)
- **HouseAge**: Median house age
- **AveRooms**: Average rooms per household
- **AveBedrms**: Average bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average occupants per household
- **Latitude**: Geographic latitude
- **Longitude**: Geographic longitude

Target variable is converted to three categories:

- **Low**: Bottom tertile of house values
- **Medium**: Middle tertile of house values
- **High**: Top tertile of house values

## 🔧 Technical Details

- **Framework**: scikit-learn, Streamlit
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Hyperparameter Tuning**: GridSearchCV with stratified cross-validation

## 📊 Evaluation Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrices for each model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Your Name - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- California Housing dataset from scikit-learn
- Streamlit for the web framework
- scikit-learn for machine learning tools
