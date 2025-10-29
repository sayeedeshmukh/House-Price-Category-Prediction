# Deployment Guide

## GitHub Setup

1. **Create a new repository on GitHub:**
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name it: `california-housing-prediction`
   - Make it public
   - Don't initialize with README (we already have one)

2. **Connect your local repository:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/california-housing-prediction.git
   git branch -M main
   git push -u origin main
   ```

## Streamlit Cloud Deployment

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy your app:**
   - Click "New app"
   - Select your repository: `california-housing-prediction`
   - Set main file path: `gui_app.py`
   - Click "Deploy!"

3. **Important Notes:**
   - Make sure `best_model.pkl` is committed to your repository
   - The app will automatically install dependencies from `requirements.txt`
   - Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (if not already done)
python house_price_model.py

# Run the Streamlit app
streamlit run gui_app.py
```

## Troubleshooting

- **Model file not found**: Make sure to run `python house_price_model.py` first
- **Dependencies issues**: Check that all packages in `requirements.txt` are compatible
- **Memory issues**: The model file might be too large; consider using compression
