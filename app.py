from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.exceptions import MlflowException

from src.data import load_data
from src.model import train_models, predict
from src.tracking import log_training_run


st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
st.title("🏠 Real Estate Investment Advisor")
st.caption("Predict property investment potential and estimated value after five years.")

with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Upload india_housing_prices.csv", type=["csv"])
    st.info("If no CSV is uploaded, a deterministic demo dataset is used.")

try:
    data, used_demo = load_data(upload)
except ValueError as error:
    st.error(str(error))
    st.stop()

if used_demo:
    st.warning("Demo data is active. Upload the project CSV for real-world predictions.")

@st.cache_resource(show_spinner="Training classification and regression models...")
def get_models(frame: pd.DataFrame):
    training_frame = frame
    if len(frame) > 10000:
        training_frame = frame.sample(10000, random_state=42)
    models = train_models(training_frame)
    try:
        log_training_run(models)
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        st.info("MLflow is not installed; predictions work, but this run was not tracked.")
    except MlflowException as error:
        st.warning(f"MLflow tracking was unavailable; predictions still work. Details: {error}")
    return models

models = get_models(data)
tab_overview, tab_predict, tab_eda = st.tabs(["Overview", "Investment Advisor", "Insights"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties", f"{len(data):,}")
    c2.metric("Average price", f"₹{data.Price_in_Lakhs.mean():,.1f} L")
    c3.metric("Classifier accuracy", f"{models.classification_accuracy:.1%}")
    c4.metric("Forecast RMSE", f"₹{models.regression_rmse:,.1f} L")
    st.subheader("Property distribution")
    counts = data.groupby("City", as_index=False).size().sort_values("size", ascending=False).head(12)
    st.plotly_chart(px.bar(counts, x="City", y="size", labels={"size": "Properties"}), use_container_width=True)
    st.dataframe(data.drop(columns=["_appreciation"], errors="ignore").head(20), use_container_width=True)

with tab_predict:
    st.subheader("Evaluate a property")
    options = {column: sorted(data[column].dropna().astype(str).unique()) for column in [
        "State", "City", "Locality", "Property_Type", "Furnished_Status", "Security",
        "Amenities", "Facing", "Owner_Type", "Availability_Status"
    ]}
    with st.form("property_form"):
        left, right = st.columns(2)
        record = {}
        for column in options:
            record[column] = left.selectbox(column.replace("_", " "), options[column])
        numeric_defaults = {
            "BHK": (int(data.BHK.median()), 1, 10), "Size_in_SqFt": (int(data.Size_in_SqFt.median()), 100, 10000),
            "Price_in_Lakhs": (float(data.Price_in_Lakhs.median()), 1.0, 100000.0),
            "Year_Built": (int(data.Year_Built.median()), 1900, 2026), "Floor_No": (int(data.Floor_No.median()), 0, 100),
            "Total_Floors": (int(data.Total_Floors.median()), 1, 100), "Nearby_Schools": (int(data.Nearby_Schools.median()), 0, 100),
            "Nearby_Hospitals": (int(data.Nearby_Hospitals.median()), 0, 100),
            "Public_Transport_Accessibility": (int(data.Public_Transport_Accessibility.median()), 0, 10),
            "Parking_Space": (int(data.Parking_Space.median()), 0, 10),
        }
        for column, (default, minimum, maximum) in numeric_defaults.items():
            record[column] = right.number_input(column.replace("_", " "), min_value=minimum, max_value=maximum, value=default)
        submitted = st.form_submit_button("Generate recommendation", type="primary")
    if submitted:
        prediction_frame = pd.DataFrame([record])
        prediction_frame["Price_per_SqFt"] = prediction_frame.Price_in_Lakhs * 100000 / prediction_frame.Size_in_SqFt.clip(lower=1)
        prediction_frame["Age_of_Property"] = max(0, pd.Timestamp.now().year - prediction_frame.Year_Built.iloc[0])
        result = predict(models, prediction_frame)
        if result["good_investment"]:
            st.success(f"Good investment candidate — model confidence {result['confidence']:.1%}")
        else:
            st.warning(f"Lower investment potential — model confidence {result['confidence']:.1%}")
        st.metric("Estimated price after 5 years", f"₹{result['price_after_5y']:,.2f} Lakhs")

with tab_eda:
    st.subheader("Exploratory insights")
    city_prices = data.groupby("City", as_index=False)["Price_in_Lakhs"].mean().sort_values("Price_in_Lakhs", ascending=False).head(15)
    st.plotly_chart(px.bar(city_prices, x="City", y="Price_in_Lakhs", title="Average price by city"), use_container_width=True)
    st.plotly_chart(px.scatter(data, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type",
                               hover_data=["City", "BHK"], title="Size and price relationship"), use_container_width=True)
    numeric = data.select_dtypes("number").drop(columns=["ID", "_appreciation"], errors="ignore")
    st.plotly_chart(px.imshow(numeric.corr(numeric_only=True), text_auto=".2f", title="Numeric feature correlation"),
                    use_container_width=True)
