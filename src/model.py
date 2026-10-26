from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CAT_FEATURES = ["State", "City", "Locality", "Property_Type", "Furnished_Status",
                "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status"]
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
                "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
                "Public_Transport_Accessibility", "Parking_Space", "Price_per_SqFt",
                "Age_of_Property"]
FEATURES = CAT_FEATURES + NUM_FEATURES


@dataclass
class TrainedModels:
    classifier: Pipeline
    regressor: Pipeline
    classification_accuracy: float
    regression_rmse: float
    regression_r2: float


def investment_label(frame: pd.DataFrame) -> pd.Series:
    """Domain label: strong appreciation and affordable entry point indicate value."""
    appreciation = frame.get("_appreciation")
    if appreciation is None:
        appreciation = (
            0.035
            + frame["Public_Transport_Accessibility"] * 0.004
            + frame["Nearby_Schools"] * 0.002
            + frame["Nearby_Hospitals"] * 0.002
            - frame["Age_of_Property"] * 0.0005
        )
    return ((appreciation >= 0.09) & (frame["Price_per_SqFt"] <= frame["Price_per_SqFt"].quantile(0.75))).astype(int)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_FEATURES),
    ])


def train_models(frame: pd.DataFrame, random_state: int = 42) -> TrainedModels:
    data = frame.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    required_target = ["Price_in_Lakhs", "Age_of_Property", "Public_Transport_Accessibility",
                       "Nearby_Schools", "Nearby_Hospitals"]
    data = data.dropna(subset=["Price_in_Lakhs"]).copy()
    for column in required_target[1:]:
        data[column] = data[column].fillna(data[column].median())
    data["Good_Investment"] = investment_label(data)
    data["Price_After_5Y"] = data["Price_in_Lakhs"] * (1 + (
        0.035 + data["Public_Transport_Accessibility"] * 0.004
        + data["Nearby_Schools"] * 0.002 + data["Nearby_Hospitals"] * 0.002
        - data["Age_of_Property"] * 0.0005
    )).clip(lower=1.05)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Price_After_5Y", "Good_Investment"])
    if len(data) < 20:
        raise ValueError("At least 20 valid property records are required for model training.")
    X, y_class, y_reg = data[FEATURES], data["Good_Investment"], data["Price_After_5Y"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=random_state, stratify=y_class
    )
    classifier = Pipeline([("preprocessor", _preprocessor()), ("model", RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
    ))])
    regressor = Pipeline([("preprocessor", _preprocessor()), ("model", RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=-1
    ))])
    classifier.fit(X_train, yc_train)
    regressor.fit(X_train, yr_train)
    class_pred = classifier.predict(X_test)
    reg_pred = regressor.predict(X_test)
    return TrainedModels(
        classifier, regressor, accuracy_score(yc_test, class_pred),
        float(np.sqrt(mean_squared_error(yr_test, reg_pred))), r2_score(yr_test, reg_pred)
    )


def predict(models: TrainedModels, record: pd.DataFrame) -> dict:
    class_probability = models.classifier.predict_proba(record[FEATURES])[0]
    prediction = int(models.classifier.predict(record[FEATURES])[0])
    forecast = float(models.regressor.predict(record[FEATURES])[0])
    return {
        "good_investment": prediction,
        "confidence": float(class_probability[prediction]),
        "price_after_5y": forecast,
    }
