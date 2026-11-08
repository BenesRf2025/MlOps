FEATURES = [
    "culture",
    "soil_moisture_pct",
    "air_temp_c",
    "air_humidity_pct",
    "solar_radiation_wm2",
    "rainfall_mm",
    "hour",
]
NUM = [f for f in FEATURES if f != "culture"]
TARGET = "irrigation_needed"


def load(path):
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df
