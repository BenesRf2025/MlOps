"""Génère des données SIMULÉES de capteurs d'irrigation (MpambolyMivoatry).

Sorties :
  data/irrigation_reference.csv : données d'entraînement (90 jours, 1 mesure/h)
  data/irrigation_drift.csv     : données "production" avec dérive (pour Evidently)

Hypothèses (simulées, à remplacer par les vraies données du pilote) :
  - climat chaud et plutôt sec type sud-ouest de Madagascar
  - cycle jour/nuit pour température, humidité de l'air, rayonnement
  - le sol se dessèche selon température/rayonnement, remonte avec pluie/irrigation
"""
import os

import numpy as np
import pandas as pd

CULTURES = ["tomate", "oignon", "mais"]
# seuil d'humidité du sol (%) sous lequel il faut irriguer, par culture
SEUILS = {"tomate": 35, "oignon": 30, "mais": 28}


def generate(n_days=90, seed=42, temp_shift=0.0, rain_factor=1.0, sensor_bias=0.0):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    ts = pd.date_range("2026-06-01", periods=n, freq="h")
    hour = ts.hour.values

    temp = 26 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1.2, n) + temp_shift
    hum = np.clip(60 - 18 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 4, n), 15, 100)
    sun = np.clip(900 * np.sin((hour - 6) / 12 * np.pi), 0, None) * rng.uniform(0.6, 1.0, n)
    sun = np.where((hour < 6) | (hour > 18), 0, sun)

    # pluie rare mais possible
    rain = np.where(rng.random(n) < 0.012 * rain_factor, rng.exponential(4, n), 0.0)

    culture = np.repeat(rng.choice(CULTURES, size=n_days), 24)

    soil = np.zeros(n)
    soil_obs = np.zeros(n)  # humidité mesurée AVANT irrigation
    water = np.zeros(n)
    needed = np.zeros(n, dtype=int)
    soil[0] = 45
    soil_obs[0] = 45
    for i in range(1, n):
        dry = 0.012 * (temp[i] - 15) + 0.0009 * sun[i] + 0.01 * (100 - hum[i])
        s = soil[i - 1] - dry + 1.8 * rain[i]
        soil_obs[i] = s
        if s < SEUILS[culture[i]]:
            needed[i] = 1
            water[i] = round(rng.uniform(8, 15), 1)  # litres
            s += water[i] * 0.9
        soil[i] = np.clip(s, 5, 80)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "culture": culture,
            "soil_moisture_pct": (np.clip(soil_obs, 5, 80) + sensor_bias + rng.normal(0, 0.4, n)).round(2),
            "air_temp_c": temp.round(2),
            "air_humidity_pct": hum.round(2),
            "solar_radiation_wm2": sun.round(1),
            "rainfall_mm": rain.round(2),
            "irrigation_needed": needed,  # cible à prédire
            "water_volume_l": water,
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    ref = generate(seed=42)
    ref.to_csv("data/irrigation_reference.csv", index=False)

    # Dérive : +4 °C, moins de pluie, capteur d'humidité qui dérive de +6 %
    drift = generate(n_days=30, seed=7, temp_shift=4.0, rain_factor=0.3, sensor_bias=6.0)
    drift["timestamp"] = drift["timestamp"] + pd.Timedelta(days=100)
    drift.to_csv("data/irrigation_drift.csv", index=False)

    print("reference:", ref.shape, "| irrigation_needed =", ref["irrigation_needed"].mean().round(3))
    print("drift    :", drift.shape, "| irrigation_needed =", drift["irrigation_needed"].mean().round(3))
