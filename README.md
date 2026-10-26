# Real Estate Investment Advisor

A Streamlit machine-learning dashboard for classifying property investment potential and forecasting estimated value after five years.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Place `india_housing_prices.csv` beside `app.py`, or upload it from the sidebar. If no file is present, the app uses a deterministic demo dataset so the dashboard remains runnable.

## Expected data

The CSV should contain the columns described in the project brief, including `State`, `City`, `Locality`, `Property_Type`, `BHK`, `Size_in_SqFt`, `Price_in_Lakhs`, `Year_Built`, furnishing/building features, amenities, ownership, and availability fields. `Price_per_SqFt` and `Age_of_Property` are derived automatically.

## Model and outputs

- Missing numeric values are median-imputed; categorical values are most-frequent-imputed and one-hot encoded.
- A balanced random forest classifies `Good Investment` using appreciation, infrastructure, age, and relative price rules.
- A random forest regressor forecasts `Price_After_5Y` in lakhs.
- The dashboard displays accuracy, RMSE, confidence, city price comparisons, size/price trends, and feature correlations.
- MLflow uses a local SQLite database (`mlflow.db`); if Windows blocks SQLAlchemy's native extension, it explicitly falls back to the local file store with the required opt-out flag.
