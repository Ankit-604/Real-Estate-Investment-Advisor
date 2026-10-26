from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
}


def _choice(rng: np.random.Generator, values: list[str], size: int) -> list[str]:
    return rng.choice(values, size=size).tolist()


def make_demo_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create realistic demo records when india_housing_prices.csv is unavailable."""
    rng = np.random.default_rng(seed)
    cities = {
        "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
        "Karnataka": ["Bengaluru", "Mysuru"],
        "Tamil Nadu": ["Chennai", "Coimbatore"],
        "Telangana": ["Hyderabad", "Warangal"],
        "Delhi": ["New Delhi", "Noida"],
    }
    states = _choice(rng, list(cities), rows)
    city = [rng.choice(cities[state]) for state in states]
    sizes = rng.integers(550, 3200, rows)
    bhk = np.clip(np.round(sizes / 650 + rng.normal(0, 0.6, rows)), 1, 5).astype(int)
    city_premium = {
        "Mumbai": 2.0, "Bengaluru": 1.5, "New Delhi": 1.7, "Pune": 1.1,
        "Chennai": 0.95, "Hyderabad": 0.9, "Noida": 0.8,
    }
    base_rate = np.array([city_premium.get(name, 0.6) for name in city])
    price_per_sqft = np.clip(base_rate * 10000 + rng.normal(0, 2200, rows), 2500, 30000)
    price = np.round(sizes * price_per_sqft / 100000, 2)
    year_built = rng.integers(1995, 2025, rows)
    school = rng.integers(0, 10, rows)
    hospitals = rng.integers(0, 6, rows)
    transport = rng.integers(1, 11, rows)
    infrastructure = (school * 0.4 + hospitals * 0.3 + transport * 0.3) / 10
    appreciation = np.clip(0.04 + infrastructure * 0.07 + rng.normal(0, 0.018, rows), 0.02, 0.2)
    return pd.DataFrame({
        "ID": np.arange(1, rows + 1),
        "State": states, "City": city,
        "Locality": [f"{name} Locality {n}" for name, n in zip(city, rng.integers(1, 9, rows))],
        "Property_Type": _choice(rng, ["Apartment", "Villa", "House"], rows),
        "BHK": bhk, "Size_in_SqFt": sizes, "Price_in_Lakhs": price,
        "Year_Built": year_built, "Furnished_Status": _choice(rng, ["Unfurnished", "Semi", "Fully"], rows),
        "Floor_No": rng.integers(0, 16, rows), "Total_Floors": rng.integers(1, 20, rows),
        "Nearby_Schools": school, "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": rng.integers(0, 3, rows),
        "Security": _choice(rng, ["Gated", "CCTV", "Guard", "None"], rows),
        "Amenities": _choice(rng, ["Gym,Pool", "Gym,Clubhouse", "Pool", "None"], rows),
        "Facing": _choice(rng, ["North", "South", "East", "West"], rows),
        "Owner_Type": _choice(rng, ["Individual", "Builder", "Agent"], rows),
        "Availability_Status": _choice(rng, ["Available", "Under Construction", "Sold"], rows),
        "_appreciation": appreciation,
    })


def load_data(uploaded_file=None, data_path: str = "india_housing_prices.csv") -> tuple[pd.DataFrame, bool]:
    """Load the user dataset, falling back to a deterministic demo dataset."""
    if uploaded_file is not None:
        frame = pd.read_csv(uploaded_file)
        used_demo = False
    else:
        path = Path(data_path)
        if path.exists():
            frame = pd.read_csv(path)
            used_demo = False
        else:
            frame = make_demo_data()
            used_demo = True
    frame = frame.drop_duplicates().copy()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    numeric = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
               "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
               "Public_Transport_Accessibility", "Parking_Space"]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            frame[column] = values.fillna(values.median())
        else:
            normalized = frame[column].astype(str).str.strip().str.lower()
            mappings = {
                "Public_Transport_Accessibility": {"low": 1, "medium": 5, "high": 10},
                "Parking_Space": {"no": 0, "yes": 1},
            }
            if column not in mappings:
                raise ValueError(f"Numeric column '{column}' contains no usable values.")
            frame[column] = normalized.map(mappings[column]).fillna(0)
    for column in sorted(REQUIRED_COLUMNS - set(numeric)):
        frame[column] = frame[column].fillna("Unknown").astype(str)
    frame["Price_per_SqFt"] = frame["Price_in_Lakhs"] * 100000 / frame["Size_in_SqFt"].clip(lower=1)
    frame["Age_of_Property"] = (pd.Timestamp.now().year - frame["Year_Built"]).clip(lower=0)
    derived_numeric = ["Price_per_SqFt", "Age_of_Property"]
    for column in numeric + derived_numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        if frame[column].isna().all():
            raise ValueError(f"Column '{column}' contains no usable numeric values.")
        frame[column] = frame[column].fillna(frame[column].median())
    return frame, used_demo
